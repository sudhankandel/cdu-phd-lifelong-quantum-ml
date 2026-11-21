import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from config import (PRIMARY_SEED, SEED, TUNING_LEARNING_RATES,
                    TUNING_EPOCHS, TUNING_BATCH_SIZES)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['classical', 'classical_matched', 'classical_mlp', 'quantum'], default='classical')
    parser.add_argument('--memory', type=int, choices=[0, 5, 10, 20, 100], default=10)
    args = parser.parse_args()
    if SEED != PRIMARY_SEED:
        parser.error('Use the primary applicant seed for tuning.')
    strategy = 'rehearsal' if args.memory else 'naive'
    combinations = list(itertools.product(TUNING_LEARNING_RATES, TUNING_EPOCHS, TUNING_BATCH_SIZES))
    results = []
    env = dict(os.environ, LL_SEED=str(SEED))
    for i, (lr, epochs, bs) in enumerate(combinations, 1):
        name = f'{args.model}_{strategy}_{args.memory}pct_validation_lr{lr:g}_bs{bs}_ep{epochs}_seed{SEED}'
        run = ROOT / 'results' / 'logs' / name
        print(f'\nTrial {i}/{len(combinations)}: {name}', flush=True)
        if not run.exists():
            subprocess.run([sys.executable, str(ROOT / 'train.py'), '--model', args.model,
                            '--strategy', strategy, '--memory', str(args.memory),
                            '--eval-mode', 'validation', '--lr', str(lr),
                            '--epochs', str(epochs), '--batch-size', str(bs)], cwd=ROOT, env=env, check=True)
        try:
            metrics = json.loads((run / 'metrics.json').read_text())
            config = json.loads((run / 'config.json').read_text())
            score = metrics['final_average_balanced_accuracy']
            assert config['seed'] == SEED and config['eval_mode'] == 'validation'
            assert config['model'] == args.model and config['memory_percent'] == args.memory
            assert config['epochs'] == epochs and config['batch_size'] == bs and config['learning_rate'] == lr
            assert not metrics['smoke_test'] and len(metrics['task_summaries']) == 3
            assert isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
        except (OSError, ValueError, KeyError, AssertionError, TypeError) as exc:
            raise RuntimeError(f'Incomplete or invalid run: {run}. Move it aside and rerun.') from exc
        results.append(dict(learning_rate=lr, epochs=epochs, batch_size=bs,
                            final_average_balanced_accuracy=score, run=name))
    results.sort(key=lambda r: (-r['final_average_balanced_accuracy'], r['epochs'], r['learning_rate'], r['batch_size']))
    out = ROOT / 'results' / 'tuning' / f'{args.model}_{strategy}_{args.memory}pct_seed{SEED}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out / 'best_config.json').write_text(json.dumps(dict(model=args.model, strategy=strategy, memory=args.memory, seed=SEED, **results[0]), indent=2))
    print('\nBest configuration:', json.dumps(results[0], indent=2))
    print('Saved to:', out)


if __name__ == '__main__':
    main()
