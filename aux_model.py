"""
Heavy memory efficiency improvements via Claude Code to ensure everything fits
in GPU RAM, due to limited computational resources.
"""

import copy
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from losses import IGNORE_INDEX, build_loss
from metrics import MetricTracker
from run_experiments import print_results_table
from training import build_dataloaders


def compute_boundary_target(labels, ignore_index=IGNORE_INDEX):
    valid = labels != ignore_index
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    h_diff = (safe[..., :, :-1] != safe[..., :, 1:]).float()
    v_diff = (safe[..., :-1, :] != safe[..., 1:, :]).float()
    r = F.pad(h_diff, (0, 1, 0, 0))
    l = F.pad(h_diff, (1, 0, 0, 0))
    d = F.pad(v_diff, (0, 0, 0, 1))
    u = F.pad(v_diff, (0, 0, 1, 0))
    boundary = ((r + l + d + u) > 0).float() * valid.float()
    return boundary.unsqueeze(1)


class BoundaryHead(nn.Module):
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class SegFormerWithBoundary(nn.Module):
    def __init__(self, base_model, boundary_hidden=64):
        super().__init__()
        self.base_model = base_model
        c1 = base_model.config.hidden_sizes[0]
        self.boundary_head = BoundaryHead(c1, hidden=boundary_hidden)

    @property
    def config(self):
        return self.base_model.config

    def forward(self, pixel_values, return_boundary=True):
        outputs = self.base_model(
            pixel_values=pixel_values, output_hidden_states=True)
        seg_logits = outputs.logits
        if not return_boundary:
            return SimpleNamespace(logits=seg_logits, boundary_logits=None)
        f1 = outputs.hidden_states[0] # (B, c1, H/4, W/4)
        boundary_logits = self.boundary_head(f1)
        return SimpleNamespace(logits=seg_logits,
                               boundary_logits=boundary_logits)


class BoundaryBCE(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX, thicken=0,
                 pos_weight_min=1.0, pos_weight_max=20.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.thicken = int(thicken)
        self.pos_weight_min = float(pos_weight_min)
        self.pos_weight_max = float(pos_weight_max)

    def forward(self, boundary_logits, labels):
        gt = compute_boundary_target(labels, self.ignore_index)
        if self.thicken > 0:
            k = 2 * self.thicken + 1
            gt = F.max_pool2d(gt, k, 1, self.thicken)

        valid = (labels != self.ignore_index).unsqueeze(1).to(
            dtype=boundary_logits.dtype)
        # BCE-with-logits is fp16-unsafe under autocast; keep it in fp32.
        with torch.autocast(device_type=boundary_logits.device.type,
                            enabled=False):
            logits_f = boundary_logits.float()
            gt_f = gt.float()
            valid_f = valid.float()
            pos = (gt_f * valid_f).sum().clamp(min=1.0)
            neg = (valid_f.sum() - pos).clamp(min=1.0)
            pw = (neg / pos).clamp(self.pos_weight_min, self.pos_weight_max)
            loss_per_pix = F.binary_cross_entropy_with_logits(
                logits_f, gt_f, pos_weight=pw, reduction='none')
            denom = valid_f.sum().clamp(min=1.0)
            return (loss_per_pix * valid_f).sum() / denom


def _upsample(logits, size):
    return F.interpolate(logits, size=size, mode='bilinear',
                         align_corners=False)


def _use_amp(device):
    return device.type == 'cuda'


def train_one_epoch_boundary(model, loader, seg_loss_fn, boundary_loss_fn,
                             optimizer, device, lambda_boundary=0.5,
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
            out = model(pixel_values=pixel_values)
            seg_logits = _upsample(out.logits, labels.shape[-2:])
            boundary_logits = _upsample(out.boundary_logits, labels.shape[-2:])
            seg_loss, parts = seg_loss_fn(seg_logits, labels)
            boundary_loss = boundary_loss_fn(boundary_logits, labels)
            loss = seg_loss + lambda_boundary * boundary_loss

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total = total + loss.detach()
        n += 1
        if (step % log_every) == 0:
            postfix = {
                'loss': f'{loss.detach().item():.3f}',
                'bnd': f'{boundary_loss.detach().item():.3f}',
            }
            for k, v in parts.items():
                postfix[k] = f'{v.item():.3f}'
            pbar.set_postfix(**postfix)
    return (total / max(n, 1)).item()


@torch.no_grad()
def evaluate_boundary(model, loader, tracker, device, max_steps=None,
                      collect_boundary=False):
    """Evaluate segmentation with the same MetricTracker as the baseline.

    The boundary head is incidental at eval time -- per the spec the
    segmentation behavior is unchanged. Set `collect_boundary=True` to
    additionally compute the mean predicted-vs-GT boundary BCE on the val
    set (useful as a sanity check).
    """
    model.eval()
    tracker.reset()
    pbar = tqdm(loader, desc='eval', dynamic_ncols=True)
    amp_enabled = _use_amp(device)
    autocast_dtype = torch.float16 if amp_enabled else torch.float32

    bnd_loss_fn = BoundaryBCE() if collect_boundary else None
    bnd_total = 0.0
    bnd_n = 0
    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=amp_enabled):
            out = model(pixel_values=pixel_values)
            seg_logits = _upsample(out.logits, labels.shape[-2:])
            if collect_boundary:
                bnd_logits = _upsample(out.boundary_logits, labels.shape[-2:])
                bnd_total += bnd_loss_fn(bnd_logits, labels).item()
                bnd_n += 1
        pred = seg_logits.argmax(dim=1)
        tracker.update(pred, labels)
    metrics = tracker.compute()
    if collect_boundary and bnd_n > 0:
        metrics = dict(metrics)
        metrics['boundary_bce'] = bnd_total / bnd_n
    return metrics


def train_and_evaluate_boundary(model, train_loader, val_loader, seg_loss_fn,
                                boundary_loss_fn, optimizer, device, tracker,
                                num_epochs=1, lambda_boundary=0.5,
                                lambda_boundary_end=None,
                                max_train_steps=None, max_eval_steps=None,
                                scaler=None,
                                save_dir=None, save_name=None):
    if scaler is None and _use_amp(device):
        scaler = torch.amp.GradScaler('cuda')
    save_each = (save_dir is not None and save_name is not None
                 and '{epoch}' in save_name)
    metrics = None
    history = {}
    for epoch in range(num_epochs):
        if lambda_boundary_end is None or num_epochs <= 1:
            cur_lambda = lambda_boundary
        else:
            t = epoch / (num_epochs - 1)
            cur_lambda = lambda_boundary + (
                lambda_boundary_end - lambda_boundary) * t
        train_loss = train_one_epoch_boundary(
            model, train_loader, seg_loss_fn, boundary_loss_fn, optimizer,
            device, lambda_boundary=cur_lambda,
            max_steps=max_train_steps, scaler=scaler)
        metrics = evaluate_boundary(model, val_loader, tracker, device,
                                    max_eval_steps)
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f}  "
              f"lambda={cur_lambda:.3f}  {metrics}")
        history[f'epoch {epoch+1}'] = metrics
        print_results_table(history)
        if save_each:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, save_name.format(epoch=epoch + 1))
            to_save = getattr(model, '_orig_mod', model)
            torch.save(to_save.state_dict(), path)
            print(f"  saved -> {path}")
    return metrics


def _maybe_compile(model, compile_model):
    if not compile_model:
        return model
    try:
        return torch.compile(model)
    except Exception as e:
        print(f"  torch.compile failed ({e}); continuing uncompiled.")
        return model


def run_boundary_experiment(train_dataset, val_dataset, processor, base_model,
                            num_classes=150, num_epochs=1, batch_size=4,
                            learning_rate=5e-5, lambda_boundary=0.5,
                            lambda_boundary_end=None,
                            boundary_thicken=0, boundary_hidden=64,
                            seg_loss_config='ce',
                            max_train_samples=None, max_val_samples=None,
                            num_workers=4, save_dir=None,
                            save_name='segformer_boundary.pt',
                            checkpoint_path=None,
                            device=None,
                            cache_dir='cache_processed',
                            compile_model=False):
    print(f"\n{'=' * 60}\n  boundary-aware ({seg_loss_config} + boundary)"
          f"\n{'=' * 60}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base = copy.deepcopy(base_model)
    model = SegFormerWithBoundary(base, boundary_hidden=boundary_hidden).to(device)

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

    # Encoder gradient checkpointing -- same memory rationale as run_one.
    base_for_ckpt = getattr(model, '_orig_mod', model)
    base_seg = getattr(base_for_ckpt, 'base_model', base_for_ckpt)
    encoder = getattr(base_seg, 'segformer', base_seg)
    if getattr(encoder, 'supports_gradient_checkpointing', False):
        encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset, processor,
        batch_size=batch_size, num_workers=num_workers,
        max_train=max_train_samples, max_val=max_val_samples,
        cache_dir=cache_dir,
        num_classes=num_classes, ignore_index=IGNORE_INDEX,
    )
    tracker = MetricTracker(num_classes=num_classes,
                            ignore_index=IGNORE_INDEX, device=device)

    seg_loss_fn = build_loss(seg_loss_config, num_classes=num_classes,
                             ignore_index=IGNORE_INDEX).to(device)
    boundary_loss_fn = BoundaryBCE(
        ignore_index=IGNORE_INDEX, thicken=boundary_thicken).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)

    metrics = train_and_evaluate_boundary(
        model, train_loader, val_loader, seg_loss_fn, boundary_loss_fn,
        optimizer, device, tracker, num_epochs=num_epochs,
        lambda_boundary=lambda_boundary,
        lambda_boundary_end=lambda_boundary_end,
        save_dir=save_dir, save_name=save_name,
    )

    # Per-epoch saves already happened above when save_name has {epoch}.
    saved_per_epoch = (save_name is not None and '{epoch}' in save_name)
    if save_dir and not saved_per_epoch:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, save_name)
        to_save = getattr(model, '_orig_mod', model)
        torch.save(to_save.state_dict(), path)
        print(f"  saved -> {path}")

    return metrics, model
