import torch
import torch.nn.functional as F


class MetricTracker:
    def __init__(self, num_classes=150, ignore_index=255,
                 biou_dilation_ratio=0.02,
                 bf1_tolerances=(3, 5, 9, 12), device=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.biou_dilation_ratio = float(biou_dilation_ratio)
        if isinstance(bf1_tolerances, int):
            bf1_tolerances = (bf1_tolerances,)
        self.bf1_tolerances = tuple(int(r) for r in bf1_tolerances)
        self.device = torch.device(device) if device is not None else None
        self.reset()

    def _dev(self):
        return self.device if self.device is not None else torch.device('cpu')

    def reset(self):
        n = self.num_classes
        dev = self._dev()
        self.confmat = torch.zeros(n, n, dtype=torch.long, device=dev)
        self.bf1_tp_p = {r: torch.zeros((), dtype=torch.float64, device=dev)
                         for r in self.bf1_tolerances}
        self.bf1_tp_r = {r: torch.zeros((), dtype=torch.float64, device=dev)
                         for r in self.bf1_tolerances}
        self.bf1_pred_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self.bf1_gt_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self.biou_inter = torch.zeros(n, dtype=torch.float64, device=dev)
        self.biou_union = torch.zeros(n, dtype=torch.float64, device=dev)

    def _ensure_device(self, device):
        if self.device is None or self.device != device:
            self.device = device
            self.confmat = self.confmat.to(device)
            for r in self.bf1_tolerances:
                self.bf1_tp_p[r] = self.bf1_tp_p[r].to(device)
                self.bf1_tp_r[r] = self.bf1_tp_r[r].to(device)
            self.bf1_pred_sum = self.bf1_pred_sum.to(device)
            self.bf1_gt_sum = self.bf1_gt_sum.to(device)
            self.biou_inter = self.biou_inter.to(device)
            self.biou_union = self.biou_union.to(device)

    @staticmethod
    def _class_boundary(mask):
        """1 where a 4-neighbor has a different class label."""
        m = mask.long()
        z = torch.zeros_like(m, dtype=torch.float32)
        br, bl = z.clone(), z.clone()
        bd, bu = z.clone(), z.clone()
        br[..., :, :-1] = (m[..., :, :-1] != m[..., :, 1:]).float()
        bl[..., :, 1:] = (m[..., :, :-1] != m[..., :, 1:]).float()
        bd[..., :-1, :] = (m[..., :-1, :] != m[..., 1:, :]).float()
        bu[..., 1:, :] = (m[..., :-1, :] != m[..., 1:, :]).float()
        return ((br + bl + bd + bu) > 0).float()

    @staticmethod
    def _erode_box(mask_bool, d):
        squeeze_back = (mask_bool.dim() == 3)
        m = mask_bool.float()
        if squeeze_back:
            m = m.unsqueeze(1)                          # (B, 1, H, W)
        m_pad = F.pad(m, (1, 1, 1, 1), value=0.0)       # 1-px zero pad
        inv_dil = F.max_pool2d(1.0 - m_pad, 2 * d + 1, 1, d)
        eroded = (1.0 - inv_dil)[:, :, 1:-1, 1:-1]
        if squeeze_back:
            eroded = eroded.squeeze(1)
        return eroded.bool()

    @torch.no_grad()
    def update(self, pred, target):
        self._ensure_device(pred.device)
        n = self.num_classes
        valid = target != self.ignore_index

        # Confusion matrix (for mIoU)
        p_flat = pred[valid].long()
        t_flat = target[valid].long()
        idx = n * t_flat + p_flat
        bc = torch.bincount(idx, minlength=n * n)[: n * n]
        self.confmat += bc.reshape(n, n)

        # Boundary extraction
        pb = self._class_boundary(pred) * valid.float()
        tb = self._class_boundary(target) * valid.float()

        # Boundary F1 (at multiple tolerances)
        self.bf1_pred_sum += pb.sum().double()
        self.bf1_gt_sum += tb.sum().double()
        pb1 = pb.unsqueeze(1)
        tb1 = tb.unsqueeze(1)
        for r in self.bf1_tolerances:
            k = 2 * r + 1
            pb_dil = F.max_pool2d(pb1, k, 1, r).squeeze(1)
            tb_dil = F.max_pool2d(tb1, k, 1, r).squeeze(1)
            self.bf1_tp_p[r] += (pb * tb_dil).sum().double()
            self.bf1_tp_r[r] += (tb * pb_dil).sum().double()

        H, W = pred.shape[-2:]
        d = max(1, int(round(self.biou_dilation_ratio * (H * H + W * W) ** 0.5)))
        sentinel = n
        pred_eff = torch.where(valid, pred, torch.full_like(pred, sentinel))
        tgt_eff = torch.where(valid, target, torch.full_like(target, sentinel))
        present = torch.unique(torch.cat([pred_eff.flatten(), tgt_eff.flatten()]))
        present = present[present < n]
        K = present.numel()
        if K > 0:
            pres_v = present.view(1, K, 1, 1)
            gt_per = (tgt_eff.unsqueeze(1) == pres_v)         # (B, K, H, W) bool
            pr_per = (pred_eff.unsqueeze(1) == pres_v)
            gt_band = gt_per & ~self._erode_box(gt_per, d)
            pr_band = pr_per & ~self._erode_box(pr_per, d)
            inter = (gt_band & pr_band).sum(dim=(0, 2, 3)).double()  # (K,)
            union = (gt_band | pr_band).sum(dim=(0, 2, 3)).double()
            self.biou_inter.index_add_(0, present.long(), inter)
            self.biou_union.index_add_(0, present.long(), union)

    def compute(self):
        confmat = self.confmat.cpu()
        diag = confmat.diag().float()
        row = confmat.sum(1).float()
        col = confmat.sum(0).float()
        union = row + col - diag
        iou = diag / union.clamp(min=1e-10)
        present = row > 0
        miou = iou[present].mean().item() if present.any() else 0.0

        ps = self.bf1_pred_sum.item()
        gs = self.bf1_gt_sum.item()
        bf1_per_tol = {}
        for r in self.bf1_tolerances:
            p = self.bf1_tp_p[r].item()
            r_ = self.bf1_tp_r[r].item()
            prec = p / (ps + 1e-10)
            rec = r_ / (gs + 1e-10)
            bf1 = (2 * prec * rec / (prec + rec + 1e-10)) if (prec + rec) > 0 else 0.0
            bf1_per_tol[f'BF1@{r}'] = bf1

        biou_inter = self.biou_inter.cpu()
        biou_union = self.biou_union.cpu()
        biou_c = biou_inter / biou_union.clamp(min=1e-10)
        present_b = biou_union > 0
        biou = biou_c[present_b].mean().item() if present_b.any() else 0.0

        return {'mIoU': miou, **bf1_per_tol, 'BIoU': biou}
