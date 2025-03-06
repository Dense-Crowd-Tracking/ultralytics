import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.loss import FocalLoss

class DOSAConLoss(nn.Module):
    def __init__(self, gamma_local=2.5, alpha=1.2, delta=1.0, tau=0.3, lambda_contrast=0.5):
        super().__init__()
        self.gamma_local = gamma_local
        self.alpha = alpha
        self.delta = delta
        self.tau = tau
        self.lambda_contrast = lambda_contrast
        self.focal_cls = FocalLoss()  # YOLO's existing focal classification loss

    def forward(self, pred_boxes, pred_cls, target_boxes, embeddings, density_map):
        # Localization: Scale-Aware Focal CIoU
        ious = bbox_iou(pred_boxes, target_boxes, CIoU=True)
        area_norm = target_boxes[..., 2] * target_boxes[..., 3]  # w*h normalized [0,1]
        saf_ciou = (1 - ious).pow(self.gamma_local) / (area_norm + 1e-7)
        
        # Density & Hardness Weights
        density_weight = 1 + self.alpha * density_map[..., None]
        hardness_weight = torch.sigmoid(5.0 * (0.5 - ious.detach()))  # Hard example mining
        
        # Contrastive Loss
        contrastive_loss = 0
        for i, j in combinations(pred_boxes.size(0)):
            if bbox_iou(pred_boxes[i], pred_boxes[j]) > self.tau:
                emb_dist = torch.norm(embeddings[i] - embeddings[j])
                contrastive_loss += torch.relu(self.delta - emb_dist).pow(2)
        
        # Combine Components
        loss_loc = (density_weight * hardness_weight * saf_ciou).mean()
        loss_contrast = self.lambda_contrast * contrastive_loss
        
        # Classification + Objectness (from YOLO)
        loss_cls = self.focal_cls(pred_cls, target_cls)
        loss_obj = ...  # Existing YOLO objectness loss
        
        return loss_loc + loss_cls + loss_obj + loss_contrast
