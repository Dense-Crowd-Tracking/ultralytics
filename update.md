#### first attempt loss
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
