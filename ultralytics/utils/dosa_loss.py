import torch
import torch.nn as nn
from itertools import combinations
from utils.metrics import bbox_iou

class DOSAConLoss(nn.Module):
    def __init__(self, gamma_local=2.5, alpha=1.2, delta=1.0, tau=0.3, lambda_contrast=0.5):
        super().__init__()
        self.gamma_local = gamma_local
        self.alpha = alpha
        self.delta = delta
        self.tau = tau
        self.lambda_contrast = lambda_contrast

    def forward(self, pred_boxes, target_boxes, embeddings, density_map):
        # Localization Loss
        ious = bbox_iou(pred_boxes, target_boxes, CIoU=True)
        area_norm = target_boxes[..., 2] * target_boxes[..., 3]
        saf_ciou = (1 - ious).pow(self.gamma_local) / (area_norm + 1e-7)
        
        # Weighting Factors
        density_weight = 1 + self.alpha * density_map[..., None]
        hardness_weight = torch.sigmoid(5.0 * (0.5 - ious.detach()))
        
        # Contrastive Loss (sampled pairs)
        n = pred_boxes.size(0)
        contrastive_loss = 0
        if n > 1:
            indices = torch.randint(0, n, (min(100, n*(n-1)//2), 2), device=pred_boxes.device)
            for i, j in indices:
                if bbox_iou(pred_boxes[i], pred_boxes[j]) > self.tau:
                    emb_dist = torch.norm(embeddings[i] - embeddings[j])
                    contrastive_loss += torch.relu(self.delta - emb_dist).pow(2)
        
        # Combine components
        loss_loc = (density_weight * hardness_weight * saf_ciou).mean()
        loss_contrast = self.lambda_contrast * contrastive_loss / (indices.size(0) + 1e-7)
        
        return loss_loc + loss_contrast