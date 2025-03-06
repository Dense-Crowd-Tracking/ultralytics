import torch
import torch.nn as nn
# from itertools import combinations
from utils.metrics import bbox_iou

class DOSAConLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1.2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_boxes, target_boxes, embeddings, density_map):
        # Scale-Aware Focal CIoU
        ious = bbox_iou(pred_boxes, target_boxes, CIoU=True)
        area_norm = target_boxes[..., 2] * target_boxes[..., 3]
        loss = (1 - ious).pow(self.gamma) / (area_norm + 1e-7)
        
        # Density weighting
        loss *= 1 + self.alpha * density_map[..., None]
        return loss.mean()