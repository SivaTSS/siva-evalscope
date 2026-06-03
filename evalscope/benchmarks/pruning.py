import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.utils.logger import get_logger

logger = get_logger()

STRATEGY = 'metadata_stratified_coreset'
DEFAULT_SEED = 20260602

PRUNING_EXTRA_PARAMS: Dict[str, Dict[str, Any]] = {
    'pruning_strategy': {
        'type': 'str',
        'description': 'Deterministic metadata/task coverage pruning strategy.',
        'value': STRATEGY,
    },
    'prune_ratio': {
        'type': 'float',
        'description': 'Fraction of the benchmark to keep when no manifest is supplied.',
        'value': 0.30,
    },
    'prune_manifest_path': {
        'type': 'str | null',
        'description': 'Optional metadata-stratified coreset manifest JSON.',
        'value': None,
    },
    'allow_partial_manifest': {
        'type': 'bool',
        'description': 'Allow manifests with IDs missing from the loaded dataset.',
        'value': False,
    },
    'seed': {
        'type': 'int',
        'description': 'Seed used for deterministic hash tie-breaking.',
        'value': DEFAULT_SEED,
    },
    'min_per_stratum': {
        'type': 'int',
        'description': 'Minimum retained examples for populated strata when possible.',
        'value': 1,
    },
}


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]


def sample_pruning_id(sample: Sample, ordinal: Optional[int] = None) -> str:
    metadata = sample.metadata or {}
    for key in ('pruning_id', 'index', 'id', 'sample_id', 'question_id'):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    if sample.id is not None:
        return str(sample.id)
    if ordinal is not None:
        return str(ordinal)
    return stable_hash({'input': sample_text(sample), 'target': sample.target, 'metadata': metadata})


def sample_text(sample: Sample) -> str:
    if isinstance(sample.input, str):
        return sample.input
    parts = []
    for message in sample.input:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                text = getattr(item, 'text', None)
                if text:
                    parts.append(str(text))
        else:
            parts.append(str(content))
    return '\n'.join(parts)


def length_bin(length: int) -> str:
    if length < 500:
        return 'xs'
    if length < 2000:
        return 'short'
    if length < 8000:
        return 'medium'
    if length < 32000:
        return 'long'
    if length < 100000:
        return 'xl'
    return 'xxl'


def count_urls(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return len([part for part in str(value or '').split(';') if part.strip()])


def domain_family(value: Any) -> str:
    if isinstance(value, list):
        urls = value
    else:
        urls = [part.strip() for part in str(value or '').split(';') if part.strip()]
    domains = []
    for url in urls:
        host = urlparse(url).netloc.lower()
        if host.startswith('www.'):
            host = host[4:]
        if host:
            domains.append(host)
    if not domains:
        return 'no_domain'
    most_common = sorted(set(domains), key=lambda host: (-domains.count(host), host))[0]
    pieces = most_common.split('.')
    return '.'.join(pieces[-2:]) if len(pieces) >= 2 else most_common


def question_type(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ('list ', 'rank ', 'alphabetical', 'order')):
        return 'enumeration'
    if any(token in lowered for token in ('calculate', 'percentage', 'difference', 'how many', 'total')):
        return 'numeric'
    if any(token in lowered for token in ('compare', 'changed', 'increase', 'decrease', 'higher', 'lower')):
        return 'comparison'
    if any(token in lowered for token in ('which ', 'identify', 'what ')):
        return 'lookup'
    return 'synthesis'


def token_bucket(text: str, buckets: int, seed: int) -> str:
    words = re.findall(r'[a-zA-Z_][a-zA-Z_0-9]+', text.lower())
    if not words:
        return 'cluster_0'
    signature = sorted(set(words[:200]))
    bucket = int(stable_hash({'words': signature, 'seed': seed}, 8), 16) % max(1, buckets)
    return f'cluster_{bucket}'


def sample_features(sample: Sample, sample_id: str, seed: int) -> Dict[str, str]:
    metadata = sample.metadata or {}
    text = sample_text(sample)
    question = str(metadata.get('question') or metadata.get('question_content') or text)
    input_tokens = metadata.get('input_tokens')
    measured_len = int(input_tokens) if str(input_tokens or '').isdigit() else len(text or question)
    url_count = count_urls(metadata.get('data_source_urls'))
    prompt_hash = metadata.get('prompt_hash')
    cluster = (
        f"cluster_{int(stable_hash({'prompt_hash': prompt_hash, 'seed': seed}, 8), 16) % 12}"
        if prompt_hash else token_bucket(question or text or sample_id, 12, seed)
    )
    return {
        'length_bin': length_bin(measured_len),
        'url_bin': 'none' if url_count == 0 else 'one' if url_count == 1 else 'few' if url_count < 6 else 'many',
        'domain_family': domain_family(metadata.get('data_source_urls')),
        'question_type': str(metadata.get('question_type') or question_type(question)),
        'difficulty': str(metadata.get('difficulty') or metadata.get('topic_difficulty') or 'unknown'),
        'platform': str(metadata.get('platform') or 'unknown'),
        'cluster': cluster,
    }


def representative_sample_ids(
    samples: Sequence[Sample],
    ratio: float,
    *,
    seed: int = DEFAULT_SEED,
    min_per_stratum: int = 1,
) -> List[str]:
    if not samples:
        return []
    keep_n = max(1, min(len(samples), int(math.ceil((len(samples) * ratio) - 1e-12))))
    rows = []
    for ordinal, sample in enumerate(samples):
        sample_id = sample_pruning_id(sample, ordinal)
        features = sample_features(sample, sample_id, seed)
        stratum_parts = [
            features['length_bin'],
            features['url_bin'],
            features['question_type'],
            features['difficulty'],
            features['platform'],
            features['cluster'],
        ]
        if features['domain_family'] != 'no_domain':
            stratum_parts.insert(2, features['domain_family'])
        stratum = tuple(stratum_parts)
        rows.append({
            'id': sample_id,
            'stratum': stratum,
            'rank': stable_hash({
                'id': sample_id,
                'stratum': stratum,
                'seed': seed
            }),
        })
    return select_rows_by_strata(rows, keep_n, min_per_stratum=min_per_stratum)


def select_rows_by_strata(rows: Sequence[Dict[str, Any]], keep_n: int, *, min_per_stratum: int = 1) -> List[str]:
    strata: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row['stratum']].append(row)
    for group in strata.values():
        group.sort(key=lambda row: row['rank'])

    selected: List[str] = []
    selected_set: Set[str] = set()
    quotas: Dict[Tuple[str, ...], int] = {}
    remainders = []
    total = len(rows)
    for stratum, group in strata.items():
        exact = len(group) * keep_n / total
        quota = int(math.floor(exact))
        if min_per_stratum > 0 and quota == 0 and len(group) > 0 and keep_n >= len(strata):
            quota = 1
        quotas[stratum] = min(quota, len(group))
        remainders.append((exact - math.floor(exact), stratum))

    while sum(quotas.values()) < keep_n:
        changed = False
        for _, stratum in sorted(remainders, reverse=True):
            if sum(quotas.values()) >= keep_n:
                break
            if quotas[stratum] < len(strata[stratum]):
                quotas[stratum] += 1
                changed = True
        if not changed:
            break
    while sum(quotas.values()) > keep_n:
        for _, stratum in sorted(remainders):
            if sum(quotas.values()) <= keep_n:
                break
            if quotas[stratum] > 0:
                quotas[stratum] -= 1

    for stratum in sorted(strata):
        for row in strata[stratum][:quotas[stratum]]:
            selected.append(row['id'])
            selected_set.add(row['id'])

    if len(selected) < keep_n:
        remaining = [row for row in rows if row['id'] not in selected_set]
        for row in sorted(remaining, key=lambda row: row['rank']):
            if len(selected) >= keep_n:
                break
            selected.append(row['id'])
    return selected


def load_manifest_ids(path: str) -> Set[str]:
    with Path(path).expanduser().open('r', encoding='utf-8') as f:
        manifest = json.load(f)
    return {str(sample_id) for sample_id in manifest.get('selected_sample_ids', [])}


class ManifestPruningMixin:
    """Filter loaded samples by metadata-stratified coreset manifest or strategy."""

    def load_dataset(self) -> DatasetDict:
        dataset = super().load_dataset()
        strategy = self.extra_params.get('pruning_strategy') or STRATEGY
        if strategy != STRATEGY:
            raise ValueError(f'Unsupported pruning_strategy={strategy!r}; expected {STRATEGY!r}.')

        manifest_path = self.extra_params.get('prune_manifest_path')
        ratio = float(self.extra_params.get('prune_ratio') or 0.30)
        seed = int(self.extra_params.get('seed') or DEFAULT_SEED)
        min_per_stratum = int(self.extra_params.get('min_per_stratum') or 1)
        allow_partial_manifest = bool(self.extra_params.get('allow_partial_manifest') or False)
        selected_ids = load_manifest_ids(manifest_path) if manifest_path else None

        pruned = {}
        for subset, subset_dataset in dataset.items():
            samples = list(subset_dataset)
            keep_ids = selected_ids
            if keep_ids is None:
                keep_ids = set(representative_sample_ids(samples, ratio, seed=seed, min_per_stratum=min_per_stratum))
            else:
                available_ids = {sample_pruning_id(sample, ordinal) for ordinal, sample in enumerate(samples)}
                missing_ids = sorted(keep_ids - available_ids)
                if missing_ids and not allow_partial_manifest:
                    raise ValueError(
                        f'Pruning manifest for {self.name}/{subset} references {len(missing_ids)} '
                        'sample IDs not present in the loaded dataset. Use the matching dataset '
                        'revision or set allow_partial_manifest=True explicitly.'
                    )
            kept = [
                copy.deepcopy(sample)
                for ordinal, sample in enumerate(samples)
                if sample_pruning_id(sample, ordinal) in keep_ids
            ]
            if not kept and samples:
                raise ValueError(f'Pruning manifest/strategy kept no samples for {self.name}/{subset}.')
            memory = MemoryDataset(kept, name=subset_dataset.name, location=subset_dataset.location)
            memory.reindex(group_size=max(1, self.repeats))
            pruned[subset] = memory
            logger.info(f'Pruned {self.name}/{subset}: {len(samples)} -> {len(kept)} samples')
        return DatasetDict(pruned)
