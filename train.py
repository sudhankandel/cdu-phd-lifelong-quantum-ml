import argparse
import warnings
import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from config import SEED, PRIMARY_SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_RATIO, get_device, set_seed
from classical_model import get_transforms, MatchedClassicalHead, MLPHead
from quantum_model import HybridQuantumResNet18

# Per-task balanced accuracy is the mean recall over the classes present in that
# task; predictions of other classes count as errors. sklearn warns about this.
warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / 'data' / 'processed' / f'seed{SEED}'
CACHE = ROOT / 'data' / 'feature_cache'
RESULTS = ROOT / 'results'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class ImageTable(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.transform = get_transforms()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        path = Path(row.image_path)
        if not path.is_absolute():
            path = ROOT / path
        with Image.open(path) as img:
            x = self.transform(img.convert('RGB'))
        return x, int(row.label)


def get_features(frame, name, device, batch_size, rebuild=False):
    """Cache deterministic ImageNet ResNet-18 features, keyed by split + seed.

    This cache must be rebuilt if image preprocessing or pretrained weights change.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    split_hash = hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()[:12]
    key = f'{name}_seed{SEED}_{split_hash}_resnet18_imagenet1k_v1_224'
    path = CACHE / f'{key}.npz'
    if path.exists() and not rebuild:
        print(f'Loading cached {name} features: {path}', flush=True)
        with np.load(path) as z:
            return torch.from_numpy(z['features'].copy()), torch.from_numpy(z['labels'].copy())

    print(f'Extracting {name} features on {device}...', flush=True)
    # Pin the pretrained weights rather than relying on a changing DEFAULT alias.
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    loader = DataLoader(ImageTable(frame), batch_size=batch_size, shuffle=False,
                        num_workers=0)
    feats, labels = [], []
    with torch.inference_mode():
        for step, (images, y) in enumerate(loader, 1):
            output = backbone(images.to(device)).cpu().float()
            feats.append(output.numpy())
            labels.append(y.numpy())
            if step % 50 == 0:
                print(f'  extracted {step * batch_size} images', flush=True)
    x = np.concatenate(feats)
    y = np.concatenate(labels).astype(np.int64)
    np.savez_compressed(path, features=x, labels=y)
    return torch.from_numpy(x), torch.from_numpy(y)


def select_rehearsal(train_df, current_task, fraction, seed):
    """Choose fraction of EACH previous class, sampled from train split only.

    The current task always uses all its training examples. At 100%, all old
    training examples are included. A minimum of one per old class is kept
    for positive fractions to avoid rounding tiny classes to zero.
    """
    current = train_df.index[train_df.task == current_task].to_numpy()
    if current_task == 1 or fraction == 0:
        return np.sort(current)
    rng = np.random.RandomState(seed + 1000 * current_task + int(fraction * 100))
    memory = []
    previous = train_df[train_df.task < current_task]
    for _, group in previous.groupby('label', sort=True):
        count = min(len(group), max(1, round(len(group) * fraction)))
        memory.extend(rng.choice(group.index.to_numpy(), size=count, replace=False).tolist())
    return np.sort(np.concatenate([current, np.array(memory, dtype=int)]))


def create_head(model_name, device, shots=None, readout=False):
    """Heads act on cached 512-d ResNet-18 features (backbone is frozen).

    classical         : Linear(512,6)                    3,078 params
    classical_matched : 512->6->pi*tanh->6x6 (no bias)   3,114 params (== quantum)
    classical_mlp     : 512->256->6                      132,870 params (reference)
    quantum           : projection + 6-qubit VQC         3,114 params
    """
    set_seed(SEED)  # identical, reproducible initialisation per experiment
    if model_name == 'classical':
        return torch.nn.Linear(512, 6).to(device), torch.float32
    if model_name == 'classical_matched':
        return MatchedClassicalHead().to(device), torch.float32
    if model_name == 'classical_mlp':
        return MLPHead().to(device), torch.float32
    if model_name == 'quantum':
        # pretrained=False: only projection + quantum head are used here.
        model = HybridQuantumResNet18(pretrained=False, shots=shots, readout=readout)
        head = torch.nn.ModuleDict({
            'projection': model.projection,
            'quantum_head': model.quantum_head,
        })
        if readout:
            head['readout'] = model.readout
        return head.to('cpu'), torch.float64
    raise ValueError(model_name)


def predict(head, model_name, x):
    if model_name != 'quantum':
        return head(x)
    angles = torch.pi * torch.tanh(head['projection'](x))
    scores = head['quantum_head'](angles)
    return head['readout'](scores) if 'readout' in head else scores


def task_accuracy(head, model_name, x, y, indices, device, dtype, batch_size):
    if len(indices) == 0:
        return None
    head.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            logits = predict(head, model_name, x[idx].to(device=device, dtype=dtype))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return float(balanced_accuracy_score(y[indices].numpy(), predictions))


def average_accuracy(R, task):
    return float(np.mean([R[task - 1][j] for j in range(task)]))


def average_forgetting(R, task):
    # Mean drop from best earlier result for previously encountered tasks.
    # No previous tasks exist after Task 1: convention = 0.
    if task == 1:
        return 0.0
    values = []
    for j in range(task - 1):
        best_previous = max(R[k][j] for k in range(j, task - 1))
        values.append(best_previous - R[task - 1][j])
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['classical', 'classical_matched', 'classical_mlp', 'quantum'],
                        required=True)
    parser.add_argument('--strategy', choices=['naive', 'rehearsal'], required=True)
    parser.add_argument('--memory', type=float, default=0.0,
                        help='Rehearsal percentage: 5, 10, 20 or 100')
    parser.add_argument('--readout', action='store_true', help='Quantum only: trainable Linear(6,6) readout')
    parser.add_argument('--class-weighted', action='store_true', help='Inverse-frequency cross-entropy weights per task')
    parser.add_argument('--shots', type=int, default=None,
                        help='Quantum only: finite shots (e.g. 1024). Default: exact.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--eval-mode', choices=['validation', 'test'], required=True,
                        help='Use validation for tuning; test only for frozen final configurations')
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Smoke test only; results NOT valid for report')
    parser.add_argument('--use-best-config', action='store_true',
                        help='Load validation-selected hyperparameters for final testing')
    args = parser.parse_args()
    best_config_path = None
    best = None
    if args.use_best_config:
        if args.eval_mode != 'test' or args.max_batches is not None:
            parser.error('--use-best-config requires full test evaluation')
        if args.readout or args.class_weighted or args.shots is not None:
            parser.error('These variants require separately tuned hyperparameters')
        if args.memory not in (0, 5, 10, 20, 100):
            parser.error('Invalid memory percentage')
        best_config_path = (RESULTS / 'tuning' /
                            f'{args.model}_{args.strategy}_{int(args.memory)}pct_seed{SEED}' /
                            'best_config.json')
        if not best_config_path.is_file():
            parser.error(f'Tuned configuration not found: {best_config_path}. Run tuning first.')
        with best_config_path.open(encoding='utf-8') as f:
            best = json.load(f)
        expected = dict(seed=SEED, model=args.model, strategy=args.strategy, memory=args.memory)
        if any(best.get(key) != value for key, value in expected.items()):
            parser.error('Tuned configuration does not match this experiment')
        args.lr = float(best['learning_rate'])
        args.epochs = int(best['epochs'])
        args.batch_size = int(best['batch_size'])
        score = float(best['final_average_balanced_accuracy'])
        if not np.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(args.lr):
            parser.error('Invalid tuned configuration')
        print(f'Loaded tuned parameters: lr={args.lr}, epochs={args.epochs}, '
              f'batch_size={args.batch_size}, validation BA={score:.6f}', flush=True)


    if SEED is None:
        parser.error('Set SEED in config.py')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    if args.strategy == 'naive' and args.memory != 0:
        parser.error('naive strategy must use --memory 0')
    if args.strategy == 'rehearsal' and args.memory not in (5, 10, 20, 100):
        parser.error('rehearsal memory must be 5, 10, 20 or 100')
    if args.shots is not None and (args.model != 'quantum' or args.shots < 1):
        parser.error('--shots is only valid for --model quantum and must be positive')
    if args.readout and args.model != 'quantum':
        parser.error('--readout is only valid for quantum')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('max-batches must be positive')
    fraction = args.memory / 100.0
    set_seed(SEED)
    # Classical model can use MPS; PennyLane simulator stays on CPU.
    device = torch.device('cpu') if args.model == 'quantum' else get_device()

    pool = pd.read_csv(SPLITS / 'train.csv')
    with (SPLITS / 'metadata.json').open(encoding='utf-8') as f:
        metadata = json.load(f)
    if metadata['seed'] != SEED:
        raise RuntimeError('Split seed differs from config.py; rerun data_preparation.py')
    if args.eval_mode == 'validation':
        train_df, eval_df = train_test_split(
            pool, test_size=VALIDATION_RATIO, random_state=SEED,
            stratify=pool['label'], shuffle=True)
        train_name, eval_name = 'tuning_train', 'validation'
    else:
        train_df = pool
        eval_df = pd.read_csv(SPLITS / 'test.csv')
        train_name, eval_name = 'train', 'test'
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    if set(train_df['image']).intersection(eval_df['image']):
        raise RuntimeError('Training/evaluation image overlap detected')
    for frame in (train_df, eval_df):
        if sorted(frame.label.unique().tolist()) != list(range(6)):
            raise RuntimeError('Expected six globally indexed classes')
    x_train, y_train = get_features(train_df, train_name, device, args.batch_size, args.rebuild_cache)
    x_eval, y_eval = get_features(eval_df, eval_name, device, args.batch_size, args.rebuild_cache)
    if not np.array_equal(y_train.numpy(), train_df.label.to_numpy()):
        raise RuntimeError('Training cache labels do not match split')
    if not np.array_equal(y_eval.numpy(), eval_df.label.to_numpy()):
        raise RuntimeError('Evaluation cache labels do not match split')

    head, dtype = create_head(args.model, device, args.shots, args.readout)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    initial_parameters = torch.cat([p.detach().cpu().flatten() for p in head.parameters()])

    suffix = f'{args.model}_{args.strategy}_{int(args.memory)}pct'
    if args.readout:
        suffix += '_readout'
    if args.class_weighted:
        suffix += '_weighted'
    if args.shots is not None:
        suffix += f'_shots{args.shots}'
    suffix += f'_{args.eval_mode}_lr{args.lr:g}_bs{args.batch_size}_ep{args.epochs}_seed{SEED}'
    if args.max_batches is not None:
        suffix += f'_SMOKE_{args.max_batches}batches'
    run_dir = RESULTS / 'logs' / suffix
    if run_dir.exists():
        raise FileExistsError(f'{run_dir} already exists; move it before rerunning')
    run_dir.mkdir(parents=True)
    configuration = {
        'seed': SEED, 'temporary_seed': SEED != PRIMARY_SEED,
        'model': args.model, 'strategy': args.strategy, 'memory_percent': args.memory,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.lr,
        'device': str(device), 'train_split_sha256': sha256(SPLITS / 'train.csv'),
        'eval_mode': args.eval_mode,
        'use_best_config': args.use_best_config,
        'best_config_source': str(best_config_path.relative_to(ROOT)) if best_config_path else None,
        'best_config_sha256': sha256(best_config_path) if best_config_path else None,
        'selected_validation_balanced_accuracy': best['final_average_balanced_accuracy'] if best else None,
        'selected_validation_run': best['run'] if best else None,
        'validation_ratio': VALIDATION_RATIO if args.eval_mode == 'validation' else None,
        'train_samples': len(train_df), 'eval_samples': len(eval_df),
        'class_to_id': metadata.get('class_to_id'), 'tasks': metadata.get('tasks'),
        'backbone': 'ResNet18_Weights.IMAGENET1K_V1',
        'preprocessing': 'Resize(224,224),ToTensor,ImageNet normalize',
        'quantum_shots': args.shots, 'readout': args.readout,
        'class_weighted': args.class_weighted,
        'weighting_scheme': 'inverse frequency of classes present in current training selection' if args.class_weighted else None,
        'trainable_parameters': n_trainable, 'python': platform.python_version(),
        'torch': torch.__version__, 'pennylane': qml.__version__,
        'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'smoke_test': args.max_batches is not None,
        'is_primary_seed': SEED == PRIMARY_SEED,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    train_df.to_csv(run_dir / 'train_split.csv', index=False)
    eval_df.to_csv(run_dir / 'eval_split.csv', index=False)
    configuration['train_pool_sha256'] = configuration['train_split_sha256']
    configuration['train_split_sha256'] = sha256(run_dir / 'train_split.csv')
    configuration['eval_split_sha256'] = sha256(run_dir / 'eval_split.csv')
    write_json(run_dir / 'config.json', configuration)
    rows, R, task_summaries = [], [[None] * 3 for _ in range(3)], []
    eval_indices = [np.flatnonzero(eval_df.task.to_numpy() == t) for t in (1, 2, 3)]
    print(f'Running {suffix} on {device}', flush=True)
    started = time.time()

    for task in (1, 2, 3):
        selected = select_rehearsal(train_df, task, fraction, SEED)
        class_weights = None
        if args.class_weighted:
            counts = torch.bincount(y_train[selected].long(), minlength=6).double()
            present = counts > 0
            class_weights = torch.ones(6, dtype=dtype, device=device)
            # Normalize present-class weights to mean 1; absent classes never
            # appear as targets in this stage and therefore need no weighting.
            inv = 1.0 / counts[present]
            class_weights[present] = (inv / inv.mean()).to(dtype=dtype, device=device)
        # Preserve head AND optimiser state across tasks: true sequential learning.
        before = torch.cat([p.detach().cpu().flatten() for p in head.parameters()])
        generator = torch.Generator().manual_seed(SEED + task)
        for epoch in range(1, args.epochs + 1):
            head.train()
            permutation = selected[torch.randperm(len(selected), generator=generator).numpy()]
            losses, gradient_norms = [], []
            for batch_number, start in enumerate(range(0, len(permutation), args.batch_size)):
                if args.max_batches is not None and batch_number >= args.max_batches:
                    break
                idx = permutation[start:start + args.batch_size]
                features = x_train[idx].to(device=device, dtype=dtype)
                targets = y_train[idx].to(device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = predict(head, args.model, features)
                loss = F.cross_entropy(logits, targets, weight=class_weights)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(grad_norm.detach().cpu()))
            row = {
                'task': task, 'epoch': epoch, 'train_samples': len(selected),
                'mean_loss': float(np.mean(losses)),
                'mean_preclip_gradient_norm': float(np.mean(gradient_norms)),
            }
            rows.append(row)
            print(f'Task {task} epoch {epoch}/{args.epochs}: loss={row["mean_loss"]:.4f}', flush=True)

        after = torch.cat([p.detach().cpu().flatten() for p in head.parameters()])
        drift = float(torch.linalg.vector_norm(after - before))
        for j in range(task):
            R[task - 1][j] = task_accuracy(head, args.model, x_eval, y_eval,
                                            eval_indices[j], device, dtype, args.batch_size)
        summary = {
            'task': task, 'train_samples': len(selected),
            'task_balanced_accuracy': R[task - 1][:task],
            'average_accuracy': average_accuracy(R, task),
            'average_balanced_accuracy': average_accuracy(R, task),
            'average_forgetting': average_forgetting(R, task),
            'parameter_drift_l2': drift,
            'drift_from_initial_l2': float(torch.linalg.vector_norm(after - initial_parameters)),
        }
        task_summaries.append(summary)
        torch.save(head.state_dict(), run_dir / f'head_after_task{task}.pt')
        print('Evaluation:', summary, flush=True)

        # Persist logs after every task, including in case later training is interrupted.
        pd.DataFrame(rows).to_csv(run_dir / 'training.csv', index=False)
        write_json(run_dir / 'metrics.json', {
            'accuracy_matrix': R, 'balanced_accuracy_matrix': R,
            'eval_mode': args.eval_mode,
        'use_best_config': args.use_best_config,
        'best_config_source': str(best_config_path.relative_to(ROOT)) if best_config_path else None,
        'best_config_sha256': sha256(best_config_path) if best_config_path else None,
        'selected_validation_balanced_accuracy': best['final_average_balanced_accuracy'] if best else None,
        'selected_validation_run': best['run'] if best else None, 'task_summaries': task_summaries,
            'final_average_balanced_accuracy': summary['average_accuracy'] if task == 3 else None,
            'elapsed_seconds': round(time.time() - started, 2),
            'smoke_test': args.max_batches is not None,
        })

    print(f'Finished. Logs: {run_dir}', flush=True)
    if args.max_batches is not None:
        print('SMOKE TEST ONLY: do not use these metrics in the assessment report.')


if __name__ == '__main__':
    main()
