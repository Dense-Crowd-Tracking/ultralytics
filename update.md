#### first attempt loss
```sh
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # Calculate object area (in pixels) from target bounding boxes (xyxy format)
        target_wh = target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2]
        object_area = torch.prod(target_wh, dim=1)  # width * height
        object_area = torch.clamp(object_area, min=1.0)  # Ensure positive area

        EPSILON = 1e-7  # Prevent division by zero
        GAMMA = 0.5     # Size impact factor

        # Calculate size factor (normalized inverse square root of area)
        size_factor = GAMMA / (torch.sqrt(object_area) + EPSILON)

        # Combine weights
        modified_weight = weight * size_factor

        # Final loss calculation
        loss_iou = ((1.0 - iou) * modified_weight).sum() / (target_scores_sum + EPSILON)

```
        
#### focal variance

```sh

    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        # Compute object area (xyxy format)
        target_wh = target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2]
        object_area = torch.prod(target_wh, dim=1)  
        object_area = torch.clamp(object_area, min=1.0)  # Prevent zero area

        EPSILON = 1e-7  
        GAMMA = 0.5  

        # Log-scaled size factor to smooth small-object weighting
        size_factor = GAMMA / (torch.log(object_area + 1) + EPSILON)  

        modified_weight = weight * size_factor

        # Apply non-linear scaling to loss (Focal Loss variant for IoU)
        ALPHA = 2.0  
        loss_iou = ((1.0 - iou) ** ALPHA * modified_weight).sum() / (weight.sum() + EPSILON)
```
        


#### dynamic update
```sh
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # Compute object area (xyxy format)
        target_wh = target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2]
        object_area = torch.prod(target_wh, dim=1)
        object_area = torch.clamp(object_area, min=1.0)  # Prevent zero area
        
        EPSILON = 1e-7  
        BASE_GAMMA = 0.5  
        
        # **Dynamic Small Object Scaling** (Adjust GAMMA based on IoU)
        gamma = BASE_GAMMA + (1 - iou.mean()).detach()  # More focus on hard examples
        
        # **Log-scaled size factor** (with gamma adaptation)
        size_factor = gamma / (torch.log(object_area + 1) + EPSILON)
        
        # **Asymptotic Confidence Weighting** (Prevent over-penalizing confident predictions)
        confidence_weight = (1.0 - iou).pow(2) / ((1.0 - iou).pow(2) + 0.5)  # Scaled focal adjustment
        
        # Combine weights
        modified_weight = weight * size_factor * confidence_weight
        
        # **Focal Loss-inspired Non-Linear Scaling**
        ALPHA = 2.0  
        loss_iou = ((1.0 - iou) ** ALPHA * modified_weight).sum() / (weight.sum() + EPSILON)
```
##### Problems and approach

###### Dynamic Gamma (gamma = BASE_GAMMA + (1 - iou.mean()))

✅ Novelty: Adaptively increases focus on hard examples as training progresses.

⚠️ Risk: May destabilize training if mean IoU fluctuates wildly.

###### Log-Scaled Size Factor

✅ Advantage: Better handles extreme object sizes vs linear scaling.

⚠️ Issue: log(object_area + 1) loses differentiation for small areas.

###### Asymptotic Confidence Weighting

✅ Strength: Prevents over-suppression of high-IoU predictions.

⚠️ Complexity: Similar to focal loss but with extra normalization.

#### normalized inverse sqaure root

```sh
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

# 1. Size-aware weight (normalized inverse sqrt)
target_wh = target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2]
object_area = torch.prod(target_wh, dim=1).clamp(min=1.0)
mean_area = object_area.mean().detach()
size_weight = torch.sqrt(mean_area / (object_area + 1e-7))

# 2. Stabilized focal weighting
FOCAL_GAMMA = 1.5  # [1-2]
focal_weight = (1.0 - iou).pow(FOCAL_GAMMA)

# 3. Combined loss
loss_iou = (focal_weight * size_weight * (1.0 - iou)).sum() / (focal_weight.sum() + 1e-7)
```

#### preserving CIOU strength
```sh
# Retain CIoU's core components
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
ciou_term = (1.0 - iou)  # Standard CIoU loss

# Our Additions (simplified)
target_wh = target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2]
object_area = torch.prod(target_wh, dim=1).clamp(min=1.0)

# 1. Size-Aware Weight (Less Aggressive)
SIZE_GAMMA = 0.3  # Reduced from 0.5
size_weight = SIZE_GAMMA / (torch.sqrt(object_area) + 1e-7)

# 2. Focal Stabilization 
FOCAL_ALPHA = 1.5  # Less than your original 2.0
focal_weight = ciou_term.detach() ** FOCAL_ALPHA  # Detach gradients

# 3. Final Hybrid Loss
loss_iou = (ciou_term * focal_weight * size_weight).sum() / (fg_mask.sum() + 1e-7)
```


#### preserving CIOU strength also divided by 99
```sh
weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
# iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
# loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
# Retain CIoU's core components
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
ciou_term = (1.0 - iou)  # Standard CIoU loss

# Your Additions (simplified)
target_wh = target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2]
object_area = torch.prod(target_wh, dim=1).clamp(min=1.0)

# 1. Size-Aware Weight (Less Aggressive)
SIZE_GAMMA = 0.3  # Reduced from 0.5
size_weight = SIZE_GAMMA / (torch.sqrt(object_area) + 1e-7)

# 2. Focal Stabilization 
FOCAL_ALPHA = 1.5  # Less than your original 2.0
focal_weight = ciou_term.detach() ** FOCAL_ALPHA  # Detach gradients

# 3. Final Hybrid Loss
loss_iou = (ciou_term * focal_weight * size_weight).sum() / (fg_mask.sum() + 1e-7)

loss_iou = loss_iou/99
```