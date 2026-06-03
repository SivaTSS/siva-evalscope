import argparse
import json

try:
    from evalscope.benchmarks.pruning_tools.pruning_core import DEFAULT_SEED, build_manifest_from_evals
except ModuleNotFoundError:
    from pruning_core import DEFAULT_SEED, build_manifest_from_evals

DEFAULTS = {
    'live_code_bench_v5': ('pass', 0.30),
    'aa_lcr': ('acc', 0.56),
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit metadata-stratified coreset pruning manifests.')
    parser.add_argument('--benchmark', required=True, choices=sorted(DEFAULTS))
    parser.add_argument('--evals', required=True, help='Directory containing predictions/ and reviews/.')
    parser.add_argument('--output', required=True, help='Output manifest directory.')
    parser.add_argument('--score-field', default=None, choices=['acc', 'pass'])
    parser.add_argument('--prune-ratio', type=float, default=None)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--min-per-stratum', type=int, default=1)
    args = parser.parse_args()

    default_score, default_ratio = DEFAULTS[args.benchmark]
    manifest = build_manifest_from_evals(
        benchmark=args.benchmark,
        evals_dir=args.evals,
        output_dir=args.output,
        score_field=args.score_field or default_score,
        ratio=args.prune_ratio if args.prune_ratio is not None else default_ratio,
        seed=args.seed,
        min_per_stratum=args.min_per_stratum,
    )
    print(
        json.dumps({
            'manifest_path': manifest['manifest_path'],
            'selected_count': manifest['selected_count']
        }, indent=2)
    )


if __name__ == '__main__':
    main()
