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
        