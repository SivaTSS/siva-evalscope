import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from evalscope.api.dataset import Sample
from evalscope.benchmarks.pruning import DEFAULT_SEED, STRATEGY, representative_sample_ids, sample_features, stable_hash


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def score_value(review_row: Dict[str, Any], score_field: str) -> float:
    value = review_row.get('sample_score', {}).get('score', {}).get('value', {})
    fallback = 'pass' if score_field == 'acc' else 'acc'
    return float(value.get(score_field, value.get(fallback, 0.0)) or 0.0)


def shipped_rows(evals_dir: str, benchmark: str, score_field: str) -> Dict[str, Dict[str, Any]]:
    evals_path = Path(evals_dir)
    prediction_files = sorted((evals_path / 'predictions').glob(f'{benchmark}__*.jsonl'))
    if not prediction_files:
        prediction_files = sorted((evals_path / 'predictions').glob(f'{benchmark}*.jsonl'))
    if not prediction_files:
        raise FileNotFoundError(f'No prediction files found for {benchmark} under {evals_path / "predictions"}')

    rows: Dict[str, Dict[str,
                         Any]] = defaultdict(lambda: {
                             'scores': {},
                             'metadata': {},
                             'task_features': {},
                             'files': []
                         })
    for pred_path in prediction_files:
        review_path = evals_path / 'reviews' / pred_path.name
        if not review_path.exists():
            raise FileNotFoundError(f'Missing review file for {pred_path.name}: {review_path}')
        model_name = pred_path.stem.split('__')[-1]
        predictions = {str(row['index']): row for row in read_jsonl(pred_path)}
        reviews = {str(row['index']): row for row in read_jsonl(review_path)}
        for index, pred_row in predictions.items():
            if index not in reviews:
                continue
            row = rows[index]
            row['scores'][model_name] = score_value(reviews[index], score_field)
            metadata = reviews[index].get('sample_score', {}).get('sample_metadata') or pred_row.get('metadata') or {}
            if metadata:
                row['metadata'].update(metadata)
            if benchmark == 'live_code_bench_v5':
                model_output = pred_row.get('model_output') or {}
                usage = model_output.get('usage') or {}
                prompt_token_ids = model_output.get('prompt_token_ids')
                if usage.get('input_tokens') is not None:
                    row['task_features'].setdefault('input_token_values', []).append(int(usage.get('input_tokens')))
                if prompt_token_ids:
                    row['task_features'].setdefault('prompt_hash_values',
                                                    []).append(stable_hash({'prompt_token_ids': prompt_token_ids}, 20))
                    row['task_features'].setdefault('prompt_token_counts', []).append(len(prompt_token_ids))
            row['files'].extend([str(pred_path), str(review_path)])
    if benchmark == 'live_code_bench_v5':
        for index, row in rows.items():
            features = row['task_features']
            input_token_values = sorted(features.get('input_token_values') or [])
            prompt_token_counts = sorted(features.get('prompt_token_counts') or [])
            prompt_hash_values = sorted(set(features.get('prompt_hash_values') or []))
            if input_token_values:
                features['input_tokens'] = int(statistics.median(input_token_values))
            if prompt_token_counts:
                features['prompt_token_count'] = int(statistics.median(prompt_token_counts))
            if prompt_hash_values:
                features['prompt_hash'] = stable_hash({'prompt_hash_values': prompt_hash_values}, 20)
            if not features.get('input_tokens') and not features.get('prompt_hash'):
                raise ValueError(f'LCB row {index} has no shipped prompt token features')
    return dict(rows)


def aa_lcr_samples(rows: Dict[str, Dict[str, Any]]) -> List[Sample]:
    samples = []
    for index in sorted(rows, key=lambda item: int(item)):
        metadata = dict(rows[index].get('metadata') or {})
        required = {'question', 'data_source_urls', 'input_tokens'}
        missing = sorted(key for key in required if key not in metadata)
        if missing:
            raise ValueError(f'AA-LCR row {index} missing metadata keys: {missing}')
        metadata['pruning_id'] = index
        samples.append(Sample(input=metadata['question'], target='', id=int(index), metadata=metadata))
    return samples


def lcb_samples(rows: Dict[str, Dict[str, Any]]) -> List[Sample]:
    samples = []
    for index in sorted(rows, key=lambda item: int(item)):
        features = dict(rows[index].get('task_features') or {})
        if not features.get('input_tokens') and not features.get('prompt_hash'):
            raise ValueError(f'LCB row {index} has no shipped prompt token features')
        metadata = {
            'pruning_id': index,
            'input_tokens': features.get('input_tokens') or features.get('prompt_token_count'),
            'prompt_hash': features.get('prompt_hash') or str(index),
            'prompt_token_count': features.get('prompt_token_count'),
            'metadata_source': 'shipped_prediction_prompt_tokens',
        }
        samples.append(Sample(input=metadata['prompt_hash'], target='', id=int(index), metadata=metadata))
    return samples


def validation(rows: Dict[str, Dict[str, Any]], selected_ids: Sequence[str]) -> Dict[str, Any]:
    selected = {str(sample_id) for sample_id in selected_ids}
    models = sorted({model for row in rows.values() for model in row['scores']})
    result = {}
    for model in models:
        full_scores = [row['scores'][model] for row in rows.values() if model in row['scores']]
        pruned_scores = [
            row['scores'][model] for index, row in rows.items() if str(index) in selected and model in row['scores']
        ]
        if not full_scores or not pruned_scores:
            continue
        full_score = statistics.mean(full_scores)
        pruned_score = statistics.mean(pruned_scores)
        result[model] = {
            'full_score': full_score,
            'pruned_score': pruned_score,
            'absolute_error': abs(full_score - pruned_score),
            'go_no_go_match_at_0_5': (full_score >= 0.5) == (pruned_score >= 0.5),
        }
    return result


def strata_summary(samples: Sequence[Sample], selected_ids: Sequence[str], seed: int) -> Dict[str, Any]:
    selected = {str(sample_id) for sample_id in selected_ids}
    full_counts: Dict[str, int] = defaultdict(int)
    selected_counts: Dict[str, int] = defaultdict(int)
    for ordinal, sample in enumerate(samples):
        sample_id = str((sample.metadata or {}).get('pruning_id', ordinal))
        features = sample_features(sample, sample_id, seed)
        key = json.dumps(features, sort_keys=True)
        full_counts[key] += 1
        if sample_id in selected:
            selected_counts[key] += 1
    return {
        'full_strata': len(full_counts),
        'selected_strata': len(selected_counts),
        'fingerprint': stable_hash({
            'full': dict(full_counts),
            'selected': dict(selected_counts)
        }, 20),
    }


def build_manifest_from_evals(
    benchmark: str,
    evals_dir: str,
    output_dir: str,
    *,
    score_field: str,
    ratio: float,
    seed: int = DEFAULT_SEED,
    min_per_stratum: int = 1,
) -> Dict[str, Any]:
    rows = shipped_rows(evals_dir, benchmark, score_field)
    selection_source = 'shipped_metadata'
    if benchmark == 'aa_lcr':
        samples = aa_lcr_samples(rows)
    elif benchmark == 'live_code_bench_v5':
        samples = lcb_samples(rows)
        selection_source = 'shipped_prediction_prompt_tokens'
    else:
        raise ValueError(f'Unsupported benchmark: {benchmark}')

    selected = representative_sample_ids(samples, ratio, seed=seed, min_per_stratum=min_per_stratum)
    manifest = {
        'benchmark': benchmark,
        'strategy': STRATEGY,
        'seed': seed,
        'requested_ratio': ratio,
        'actual_ratio': len(selected) / max(1, len(samples)),
        'selected_sample_ids': selected,
        'source_eval_files': sorted({file
                                     for row in rows.values()
                                     for file in row['files']}),
        'score_field': score_field,
        'sample_count': len(samples),
        'selected_count': len(selected),
        'selection_basis': 'metadata_text_task_coverage_only',
        'selection_source': selection_source,
        'strata_summary': strata_summary(samples, selected, seed),
        'validation': validation(rows, selected),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output_dir) / f'{benchmark}_{STRATEGY}_{len(selected)}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    manifest['manifest_path'] = str(output_path)
    return manifest
