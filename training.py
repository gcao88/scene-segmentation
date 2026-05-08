"""
Heavy memory efficiency improvements via Claude Code to ensure everything fits
in GPU RAM, due to limited computational resources.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from losses import IGNORE_INDEX

torch.backends.cudnn.benchmark = True

def _silence_dataloader_shutdown():
    try:
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
    except ImportError:
        return
    _orig = _MultiProcessingDataLoaderIter.__del__

    def _safe_del(self):
        try:
            _orig(self)
        except (AssertionError, Exception):
            pass

    _MultiProcessingDataLoaderIter.__del__ = _safe_del


_silence_dataloader_shutdown()


class SegmentationDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_samples=None,
                 cache_dir=None,
                 num_classes=150, ignore_index=IGNORE_INDEX):
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        self.dataset = hf_dataset
        self.processor = processor
        self.cache_dir = cache_dir
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def _process(self, idx):
        if self.cache_dir:
            path = os.path.join(self.cache_dir, f'sample_{idx}.pt')
            if os.path.exists(path):
                try:
                    return torch.load(path, weights_only=True)
                except Exception:
                    # Corrupt cache (e.g. prior crash mid-write). Remove and redo.
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        sample = self.dataset[idx]
        image = sample['image'].convert('RGB')
        mask = sample['annotation']
        enc = self.processor(images=image, segmentation_maps=mask,
                             return_tensors='pt')
        out = {
            'pixel_values': enc['pixel_values'].squeeze(0),
            'labels': enc['labels'].squeeze(0).long(),
        }
        if self.cache_dir:
            final = os.path.join(self.cache_dir, f'sample_{idx}.pt')
            # Atomic write: tmp per-pid path + rename. Avoids half-written files
            # if a worker is killed (e.g. by OOM) mid-save.
            tmp = f'{final}.tmp.{os.getpid()}'
            try:
                torch.save(out, tmp)
                os.replace(tmp, final)
            except Exception:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        return out

    def __getitem__(self, idx):
        return self._process(idx)


def build_dataloaders(train_ds, val_ds, processor, batch_size=4, num_workers=4,
                      max_train=None, max_val=None, cache_dir=None,
                      num_classes=150,
                      ignore_index=IGNORE_INDEX):
    train_cache = os.path.join(cache_dir, 'train') if cache_dir else None
    val_cache = os.path.join(cache_dir, 'val') if cache_dir else None

    train = SegmentationDataset(train_ds, processor, max_samples=max_train,
                                cache_dir=train_cache,
                                num_classes=num_classes,
                                ignore_index=ignore_index)
    val = SegmentationDataset(val_ds, processor, max_samples=max_val,
                              cache_dir=val_cache,
                              num_classes=num_classes,
                              ignore_index=ignore_index)

    # pin_memory only helps when we'll transfer to CUDA.
    loader_kwargs = dict(pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        # persistent_workers=True causes noisy GC tracebacks in Jupyter/Colab
        # and only helps across epochs (no benefit for single-epoch sweeps).
        loader_kwargs.update(prefetch_factor=4)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, **loader_kwargs)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, **loader_kwargs)
    return train_loader, val_loader


def _upsample(logits, size):
    return F.interpolate(logits, size=size, mode='bilinear', align_corners=False)


def _use_amp(device):
    return device.type == 'cuda'


def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    max_steps=None, scaler=None, log_every=20):
    model.train()
    total = torch.zeros((), device=device)
    n = 0
    pbar = tqdm(loader, desc='train', dynamic_ncols=True)
    amp_enabled = _use_amp(device)
    autocast_dtype = torch.float16 if amp_enabled else torch.float32

    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=amp_enabled):
            logits = model(pixel_values=pixel_values).logits
            logits = _upsample(logits, labels.shape[-2:])
            loss, parts = loss_fn(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total = total + loss.detach()
        n += 1

        # Keep the progress bar useful without forcing a sync every step.
        if (step % log_every) == 0:
            pbar.set_postfix(
                loss=f'{loss.detach().item():.3f}',
                **{k: f'{v.item():.3f}' for k, v in parts.items()},
            )
    return (total / max(n, 1)).item()


@torch.no_grad()
def evaluate(model, loader, tracker, device, max_steps=None):
    model.eval()
    tracker.reset()
    pbar = tqdm(loader, desc='eval', dynamic_ncols=True)
    amp_enabled = _use_amp(device)
    autocast_dtype = torch.float16 if amp_enabled else torch.float32
    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=amp_enabled):
            logits = model(pixel_values=pixel_values).logits
            logits = _upsample(logits, labels.shape[-2:])
        pred = logits.argmax(dim=1)
        tracker.update(pred, labels)
    return tracker.compute()


def train_and_evaluate(model, train_loader, val_loader, loss_fn, optimizer,
                       device, tracker, num_epochs=1,
                       max_train_steps=None, max_eval_steps=None,
                       scaler=None):
    if scaler is None and _use_amp(device):
        scaler = torch.amp.GradScaler('cuda')
    metrics = None
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     device, max_train_steps, scaler=scaler)
        metrics = evaluate(model, val_loader, tracker, device, max_eval_steps)
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f}  {metrics}")
    return metrics
