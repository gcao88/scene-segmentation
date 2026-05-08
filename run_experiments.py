"""
Heavy memory efficiency improvements via Claude Code to ensure everything fits
in GPU RAM, due to limited computational resources.
"""

import copy
import gc
import os
import torch

from torch.utils.data import DataLoader

from losses import build_loss, IGNORE_INDEX
from metrics import MetricTracker
from training import (
    SegmentationDataset, build_dataloaders, train_and_evaluate, evaluate,
)


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


CONFIGS = [
    'baseline',
    'ce',
    'ce+bdou',
]


def _maybe_compile(model, compile_model):
    if not compile_model:
        return model
    try:
        return torch.compile(model)
    except Exception as e:
        print(f"  torch.compile failed ({e}); continuing uncompiled.")
        return model


def run_one(config_name, base_model, train_dataset, val_dataset, processor,
            device, num_classes=150, num_epochs=1, batch_size=4,
            learning_rate=5e-5, max_train_samples=None, max_val_samples=None,
            num_workers=4, save_dir=None, save_name=None,
            checkpoint_path=None, cache_dir=None, compile_model=False):
    print(f"\n{'=' * 60}\n  {config_name}\n{'=' * 60}")

    # `base_model` is kept on CPU by `run_all`; deepcopy then `.to(device)`
    # so we never hold two GPU copies at once.
    model = copy.deepcopy(base_model).to(device)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device,
                                weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [resume] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"  [resume] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        print(f"  [resume] loaded weights from {checkpoint_path}")

    model = _maybe_compile(model, compile_model)

    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset, processor,
        batch_size=batch_size, num_workers=num_workers,
        max_train=max_train_samples, max_val=max_val_samples,
        cache_dir=cache_dir,
        num_classes=num_classes, ignore_index=IGNORE_INDEX,
    )
    tracker = MetricTracker(num_classes=num_classes, ignore_index=IGNORE_INDEX,
                            device=device)

    if config_name == 'baseline':
        metrics = evaluate(model, val_loader, tracker, device)
        print(f"  {metrics}")
        return metrics

    # Encoder gradient checkpointing: at batch_size=16, 512x512 fp16, the
    # SegFormer encoder's saved activations alone ate ~20 GB on the 22 GB
    # GPU, leaving no room for any boundary loss's intermediate tensors.
    # Recomputing encoder forward during backward trades ~25% wall-clock for
    # most of that activation memory back. SegFormer's HF impl gates on
    # self.training, so eval/baseline is unaffected.
    #
    # The `SegformerForSemanticSegmentation` wrapper itself doesn't support
    # checkpointing — only the inner `SegformerModel` encoder does — so we
    # target `.segformer` directly.
    base_for_ckpt = getattr(model, '_orig_mod', model)
    encoder = getattr(base_for_ckpt, 'segformer', base_for_ckpt)
    if getattr(encoder, 'supports_gradient_checkpointing', False):
        encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    loss_fn = build_loss(config_name, num_classes=num_classes,
                         ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)

    metrics = train_and_evaluate(
        model, train_loader, val_loader, loss_fn, optimizer, device, tracker,
        num_epochs=num_epochs,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            safe_name = config_name.replace('+', '_')
            save_name = f'segformer_{safe_name}.pt'
        path = os.path.join(save_dir, save_name)
        # Compiled models wrap the real one under `_orig_mod`.
        to_save = getattr(model, '_orig_mod', model)
        torch.save(to_save.state_dict(), path)
        print(f"  saved -> {path}")

    return metrics


def _build_val_loader(val_dataset, processor, batch_size, num_workers,
                      max_val_samples, cache_dir, num_classes):
    val_cache = os.path.join(cache_dir, 'val') if cache_dir else None
    val = SegmentationDataset(val_dataset, processor,
                              max_samples=max_val_samples,
                              cache_dir=val_cache,
                              num_classes=num_classes,
                              ignore_index=IGNORE_INDEX)
    loader_kwargs = dict(pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=4)
    return DataLoader(val, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, **loader_kwargs)


def evaluate_checkpoint(checkpoint_path, base_model, val_dataset, processor,
                        device=None, num_classes=150, batch_size=4,
                        max_val_samples=None, num_workers=4,
                        cache_dir='cache_processed'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = copy.deepcopy(base_model).to(device)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)

    val_loader = _build_val_loader(val_dataset, processor, batch_size,
                                   num_workers, max_val_samples, cache_dir,
                                   num_classes)
    tracker = MetricTracker(num_classes=num_classes, ignore_index=IGNORE_INDEX,
                            device=device)
    metrics = evaluate(model, val_loader, tracker, device)

    del model
    _cleanup()
    return metrics


def evaluate_all_checkpoints(checkpoint_dir, base_model, val_dataset, processor,
                             configs=None, include_baseline=True, **kwargs):
    if configs is None:
        configs = [c for c in CONFIGS if c != 'baseline']
    results = {}
    if include_baseline:
        print(f"\n{'=' * 60}\n  baseline (no checkpoint)\n{'=' * 60}")
        results['baseline'] = evaluate_checkpoint(
            None, base_model, val_dataset, processor, **kwargs)
        print(f"  {results['baseline']}")
    for name in configs:
        safe = name.replace('+', '_')
        path = os.path.join(checkpoint_dir, f'segformer_{safe}.pt')
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        print(f"\n{'=' * 60}\n  {name} <- {path}\n{'=' * 60}")
        results[name] = evaluate_checkpoint(
            path, base_model, val_dataset, processor, **kwargs)
        print(f"  {results[name]}")
    print_results_table(results)
    return results


def print_results_table(results):
    if not results:
        return
    metric_keys = list(next(iter(results.values())).keys())
    name_w = 22
    col_w = 10
    total_w = name_w + len(metric_keys) * (col_w + 1)
    header = f"{'Configuration':<{name_w}}" + ''.join(
        f' {k:>{col_w}}' for k in metric_keys)
    print('\n' + '=' * total_w)
    print(header)
    print('-' * total_w)
    for name, m in results.items():
        row = f"{name:<{name_w}}" + ''.join(
            f' {m[k]:>{col_w}.4f}' for k in metric_keys)
        print(row)
    print('=' * total_w)


def run_all(train_dataset, val_dataset, processor, base_model,
            configs=None, num_classes=150, num_epochs=1, batch_size=4,
            learning_rate=5e-5, max_train_samples=None, max_val_samples=None,
            num_workers=4, save_dir=None, save_name=None,
            checkpoint_path=None, device=None,
            cache_dir='cache_processed', compile_model=False):
    """
    Train and evaluate every configuration. Returns a dict:
        {config_name: {'mIoU': ..., 'BF1@3': ..., 'BF1@5': ..., 'BF1@9': ...,
                       'BF1@12': ..., 'BIoU': ...}}

    `cache_dir` holds processed (pixel_values, labels) tensors on disk so
    the HuggingFace processor only runs once across all six configs. Pass
    `cache_dir=None` to disable.

    `compile_model=True` applies `torch.compile` (often 10-30% on T4, but
    flaky on Windows; the code falls back gracefully if it fails).
    """
    if configs is None:
        configs = CONFIGS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    # Keep base_model on CPU so deepcopy in run_one never peaks at 2x model
    # memory on GPU.
    base_model.cpu()
    _cleanup()

    # `save_name` and `checkpoint_path` only make sense for a single config
    # (a multi-config sweep wants the default per-config naming/no resume).
    if (save_name is not None or checkpoint_path is not None) and len(configs) > 1:
        raise ValueError(
            'save_name and checkpoint_path are only supported when running a '
            'single config; got configs=%r' % configs)

    for name in configs:
        metrics = run_one(
            name, base_model, train_dataset, val_dataset, processor, device,
            num_classes=num_classes, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            num_workers=num_workers, save_dir=save_dir,
            save_name=save_name, checkpoint_path=checkpoint_path,
            cache_dir=cache_dir, compile_model=compile_model,
        )
        results[name] = metrics
        del metrics
        print_results_table(results)  # incremental so you can see progress
        _cleanup()  # free GPU memory held by the just-finished run
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  [gpu] {(total-free)/1e9:.2f}/{total/1e9:.2f} GB used after {name}")
    return results
