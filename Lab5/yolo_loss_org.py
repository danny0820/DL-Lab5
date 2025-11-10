import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: predictions after sigmoid, shape [N, *]
        targets: ground truth labels (0 or 1), shape [N, *]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type

    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: [bsz, grid, grid, anchors, 4] (raw predictions)
        target_boxes: [bsz, grid, grid, anchors, 4] (encoded targets)
        anchors: list of (w, h) for the anchors at this scale (normalized 0-1)
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        anchors = torch.tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)

        # coordinate offset for each grid cell
        grid_range = torch.arange(grid, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        grid_x = grid_x.view(1, grid, grid, 1, 1)
        grid_y = grid_y.view(1, grid, grid, 1, 1)

        if self.type == 'giou':
            ##################YOUR CODE HERE##########################
            # 1. predicted centre (cell offset) and size
            # 2. target centre still stored as cell offset, convert to same system
            # 3. Convert both to image-normalised coordinates
            # 4. boxes to corner format
            # 5. Intersection box
            # 6. union area
            # 7. smallest enclosing box
            ##########################################################
            giou = iou - (c_area - union) / (c_area + eps)
            giou_loss = 1.0 - giou
            
            return giou_loss.squeeze(-1)

        elif self.type == 'mse':
            ##################YOUR CODE HERE##########################
            #### MSE box loss ####
            ## hints: decode predicted boxes, compute MSE loss on box coordinates
            ## Don't forget to caculate w, h mse in log space.
            ##########################################################
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
        lambda_class=1.0,
        anchors=None,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.box_loss = BoxLoss(loss_type='giou')
        self.anchors = anchors  # List of anchor boxes per scale
    # Check for NaNs in any of the loss scalars and print which one is NaN
    
    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [batch, grid, grid, 75]
        targets: list of 3 scales, each [batch, grid, grid, 3, 25]
        """
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss_pos = torch.tensor(0.0, device=device)
        total_obj_loss_neg = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0
        total_num_neg = 0
        batch_size = predictions[0].size(0)

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            bsz, grid, _, num_anchors, _ = gt.shape
            # Reshape prediction: [B, H, W, 75] -> [B, H, W, 3, 25]
            pred = pred.view(bsz, grid, grid, num_anchors, -1)
            ##################YOUR CODE HERE##########################
            ##########################################################
            pass
            ##########################################################
            ##########################################################
            

        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)

        total_box_loss = total_box_loss / pos_denom
        total_obj_loss = total_obj_loss_pos / pos_denom
        total_cls_loss = total_cls_loss / pos_denom
        total_noobj_loss = total_obj_loss_neg / neg_denom

        # Combined loss
        
        total_loss = (
            self.lambda_coord * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_cls_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'box': total_box_loss,
            'obj': total_obj_loss,
            'noobj': total_noobj_loss,
            'cls': total_cls_loss,
        }
        
        return loss_dict
