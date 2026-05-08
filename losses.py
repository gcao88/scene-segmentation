import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = 255


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class BDoULoss(nn.Module):
    def __init__(self, num_classes=150, ignore_index=IGNORE_INDEX,
                 dilation=5, eps=1e-6, class_chunk=16):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dilation = dilation
        self.eps = eps
        self.class_chunk = class_chunk

    @staticmethod
    def _chunk_terms(logits, ids_chunk, log_norm, safe_u, valid_mask, k, p):
        logits_slice = logits.index_select(1, ids_chunk)
        q = (logits_slice - log_norm).exp().to(dtype=logits_slice.dtype)
        ids = ids_chunk.view(1, -1, 1, 1)
        g = (safe_u == ids).to(dtype=q.dtype) * valid_mask
        dilated = F.max_pool2d(g, k, stride=1, padding=p)
        eroded = -F.max_pool2d(-g, k, stride=1, padding=p)
        band = (dilated - eroded) * valid_mask
        num = ((q - g).abs() * band).sum(dtype=torch.float32)
        den = (torch.max(q, g) * band).sum(dtype=torch.float32)
        return num, den

    def forward(self, logits, targets):
        log_norm = torch.logsumexp(logits.float(), dim=1, keepdim=True).to(logits.dtype)

        valid = targets != self.ignore_index
        safe = torch.where(valid, targets, torch.zeros_like(targets))
        valid_mask = valid.unsqueeze(1).to(dtype=logits.dtype)
        safe_u = safe.unsqueeze(1)

        # Iterate only over present classes; absent ones contribute zero.
        present = torch.unique(targets)
        present = present[present != self.ignore_index]
        K = present.numel()

        k, p = self.dilation, self.dilation // 2
        num = logits.new_zeros((), dtype=torch.float32)
        den = logits.new_zeros((), dtype=torch.float32)
        use_ckpt = self.training and logits.requires_grad
        for i in range(0, K, self.class_chunk):
            ids_chunk = present[i:i + self.class_chunk]
            if use_ckpt:
                num_c, den_c = checkpoint(
                    self._chunk_terms,
                    logits, ids_chunk, log_norm, safe_u, valid_mask, k, p,
                    use_reentrant=False,
                )
            else:
                num_c, den_c = self._chunk_terms(
                    logits, ids_chunk, log_norm, safe_u, valid_mask, k, p)
            num = num + num_c
            den = den + den_c
        return num / (den + self.eps)


class CombinedLoss(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.modules_ = nn.ModuleDict({n: m for n, (m, _) in components.items()})
        self.weights = {n: float(w) for n, (_, w) in components.items()}

    def forward(self, logits, targets):
        total = 0.0
        parts = {}
        for name, module in self.modules_.items():
            v = module(logits, targets)
            parts[name] = v.detach()
            total = total + self.weights[name] * v
        return total, parts


def build_loss(config, num_classes=150, ignore_index=IGNORE_INDEX):
    ce = lambda: CrossEntropyLoss(ignore_index)
    if config == 'ce':
        return CombinedLoss({'ce': (ce(), 1.0)})
    if config == 'ce+bdou':
        return CombinedLoss({
            'ce': (ce(), 1.0),
            'bdou': (BDoULoss(num_classes, ignore_index), 1.0),
        })
    if config == 'bdou':
        return CombinedLoss({
            'bdou': (BDoULoss(num_classes, ignore_index), 1.0),
        })
    raise ValueError(f"Unknown loss config: {config}")
