import torch
import torch.nn as nn
from .metrics import bbox_iou

class DOSAConLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=1.5):
        super().__init__()
        self.gamma = gamma  # Focuses loss on small objects (higher = more focus)
        self.alpha = alpha  # Density importance (1.0-2.0 works best)

    def forward(self, pred_boxes, target_boxes):
        # 1. Calculate CIoU
        iou = bbox_iou(pred_boxes, target_boxes, CIoU=True)
        
        # 2. Scale-aware weighting (inverse area normalization)
        target_areas = target_boxes[..., 2] * target_boxes[..., 3]  # w*h
        scale_weight = 1 / (target_areas + 1e-7)  # Smaller objects get higher weight
        
        # 3. Density weighting (simple count-based)
        density_map = self._fast_density_map(target_boxes)
        density_weight = 1 + self.alpha * density_map[..., None]
        
        # 4. Final loss
        loss = (1 - iou).pow(self.gamma) * scale_weight * density_weight
        return loss.mean()

    def _fast_density_map(self, boxes, grid_size=32):
        """Creates low-res density map (faster computation)"""
        density = torch.zeros(1, grid_size, grid_size, device=boxes.device)
        grid_x = (boxes[..., 0] * grid_size).long().clamp(0, grid_size-1)
        grid_y = (boxes[..., 1] * grid_size).long().clamp(0, grid_size-1)
        for x, y in zip(grid_x, grid_y):
            density[0, y, x] += 1  # Count objects per grid cell
        return density / density.max()