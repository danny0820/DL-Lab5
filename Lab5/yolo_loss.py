import torch
import torch.nn as nn
import torch.nn.functional as F

# FocalLoss類別實現了焦點損失，用於處理類別不平衡問題
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 控制正負樣本的權重
        self.gamma = gamma  # 控制難易樣本的權重
        self.reduction = reduction  # 損失的縮減方式

    def forward(self, inputs, targets):
        """
        inputs: 預測值，經過sigmoid後的輸出，形狀為[N, *]
        targets: 真實標籤（0或1），形狀為[N, *]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)  # 將輸入轉換為概率
        p_t = targets * probs + (1 - targets) * (1 - probs)  # 計算正負樣本的概率
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)  # 計算alpha權重
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss  # 計算焦點損失

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# BoxLoss類別實現了邊界框損失，支持GIoU和MSE兩種方式
class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type  # 損失類型，可選'giou'或'mse'

    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: 預測的邊界框，形狀為[bsz, grid, grid, anchors, 4]
        target_boxes: 真實的邊界框，形狀為[bsz, grid, grid, anchors, 4]
        anchors: 該尺度下的錨框列表，形狀為(w, h)
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        anchors = torch.tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)

        # 計算每個網格單元的偏移量
        grid_range = torch.arange(grid, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        grid_x = grid_x.view(1, grid, grid, 1, 1)
        grid_y = grid_y.view(1, grid, grid, 1, 1)

        if self.type == 'giou':
            #------------------------------------------------------
            eps = 1e-9
            # 預測框的中心點和大小
            pred_xy = torch.sigmoid(pred_boxes[..., 0:2])  # 預測的中心點
            pred_wh = torch.exp(pred_boxes[..., 2:4]) * anchors  # 預測的寬高

            # 真實框的中心點和大小
            target_xy = target_boxes[..., 0:2]
            target_wh = target_boxes[..., 2:4]

            # 將中心點轉換為圖像歸一化坐標
            pred_xy_norm = (pred_xy + torch.cat([grid_x, grid_y], dim=-1)) / grid
            target_xy_norm = (target_xy + torch.cat([grid_x, grid_y], dim=-1)) / grid

            # 將框轉換為角點格式
            pred_x1y1 = pred_xy_norm - pred_wh / 2
            pred_x2y2 = pred_xy_norm + pred_wh / 2
            target_x1y1 = target_xy_norm - target_wh / 2
            target_x2y2 = target_xy_norm + target_wh / 2

            # 計算交集框
            inter_x1y1 = torch.max(pred_x1y1, target_x1y1)
            inter_x2y2 = torch.min(pred_x2y2, target_x2y2)
            inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
            inter_area = inter_wh[..., 0] * inter_wh[..., 1]

            # 計算並集面積
            pred_area = pred_wh[..., 0] * pred_wh[..., 1]
            target_area = target_wh[..., 0] * target_wh[..., 1]
            union = pred_area + target_area - inter_area + eps

            iou = inter_area / union

            # 計算最小外接框
            c_x1y1 = torch.min(pred_x1y1, target_x1y1)
            c_x2y2 = torch.max(pred_x2y2, target_x2y2)
            c_wh = c_x2y2 - c_x1y1
            c_area = c_wh[..., 0] * c_wh[..., 1] + eps
            # ------------------------------------------------------
            giou = iou - (c_area - union) / (c_area + eps)
            giou_loss = 1.0 - giou

            # [B,S,S,A]
            return giou_loss

        elif self.type == 'mse':
            # 均方誤差損失
            #------------------------------------------------------
            pred_xy = torch.sigmoid(pred_boxes[..., 0:2])
            pred_wh = pred_boxes[..., 2:4]

            target_xy = target_boxes[..., 0:2]
            target_wh = target_boxes[..., 2:4]

            xy_loss = F.mse_loss(pred_xy, target_xy, reduction='none')
            wh_loss = F.mse_loss(pred_wh, target_wh, reduction='none')

            box_loss = xy_loss.sum(dim=-1) + wh_loss.sum(dim=-1)

            return box_loss
        #------------------------------------------------------
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")

# YOLOv3Loss類別實現了YOLOv3的損失函數，包含邊界框損失、物體性損失和類別損失
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
        self.lambda_coord = lambda_coord  # 邊界框損失的權重
        self.lambda_obj = lambda_obj  # 物體性損失的權重
        self.lambda_noobj = lambda_noobj  # 無物體損失的權重
        self.lambda_class = lambda_class  # 類別損失的權重

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.box_loss = BoxLoss(loss_type='giou')
        self.anchors = anchors  # 每個尺度的錨框列表

    def forward(self, predictions, targets):
        """
        predictions: 包含3個尺度的預測，每個尺度形狀為[batch, grid, grid, 75]
        targets: 包含3個尺度的真實標籤，每個尺度形狀為[batch, grid, grid, 3, 25]
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
            pred = pred.view(bsz, grid, grid, num_anchors, -1)
            ###################YOUR CODE HERE#########################
            # obj_mask = gt[..., 4] > 0
            # noobj_mask = gt[..., 4] == 0

            # pred_boxes = pred[..., 0:4]
            # pred_conf = pred[..., 4]
            # pred_cls = pred[..., 5:]

            # target_boxes = gt[..., 0:4]
            # target_conf = gt[..., 4]
            # target_cls = gt[..., 5:]

            # num_pos = obj_mask.sum().item()
            # num_neg = noobj_mask.sum().item()
            # total_num_pos += num_pos
            # total_num_neg += num_neg

            # if num_pos > 0:
            #     # 使用 BoxLoss 計算邊界框損失
            #     box_loss = self.box_loss(pred_boxes, target_boxes, anchors)
            #     box_loss_pos = box_loss[obj_mask]
            #     total_box_loss += box_loss_pos.sum()

            #     # 計算 IoU 用於 objectness target
            #     with torch.no_grad():
            #         pred_boxes_pos = pred_boxes[obj_mask]
            #         target_boxes_pos = target_boxes[obj_mask]
                    
            #         pos_indices = obj_mask.nonzero(as_tuple=False)
            #         anchor_indices = pos_indices[:, 3]
                    
            #         anchor_tensor = torch.tensor(anchors, device=device, dtype=pred_boxes_pos.dtype)
            #         anchors_for_pos = anchor_tensor[anchor_indices]
                    
            #         eps = 1e-9
            #         pred_xy = torch.sigmoid(pred_boxes_pos[:, 0:2])
            #         pred_wh = torch.exp(pred_boxes_pos[:, 2:4]) * anchors_for_pos
                    
            #         target_xy = target_boxes_pos[:, 0:2]
            #         target_wh = target_boxes_pos[:, 2:4]
                    
            #         grid_coords = pos_indices[:, 1:3].float()
            #         grid_x = grid_coords[:, 1:2]
            #         grid_y = grid_coords[:, 0:1]
                    
            #         pred_xy_norm = (pred_xy + torch.cat([grid_x, grid_y], dim=1)) / grid
            #         target_xy_norm = (target_xy + torch.cat([grid_x, grid_y], dim=1)) / grid
                    
            #         pred_x1y1 = pred_xy_norm - pred_wh / 2
            #         pred_x2y2 = pred_xy_norm + pred_wh / 2
            #         target_x1y1 = target_xy_norm - target_wh / 2
            #         target_x2y2 = target_xy_norm + target_wh / 2
                    
            #         inter_x1y1 = torch.max(pred_x1y1, target_x1y1)
            #         inter_x2y2 = torch.min(pred_x2y2, target_x2y2)
            #         inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
            #         inter_area = inter_wh[:, 0] * inter_wh[:, 1]
                    
            #         pred_area = pred_wh[:, 0] * pred_wh[:, 1]
            #         target_area = target_wh[:, 0] * target_wh[:, 1]
            #         union = pred_area + target_area - inter_area + eps
            #         iou = inter_area / union
            #         ious = iou.clamp(0, 1)

            #     # 使用 FocalLoss 計算 objectness 損失（正樣本）
            #     pred_conf_pos = pred_conf[obj_mask]
            #     obj_loss = self.focal_loss(pred_conf_pos, ious)
            #     total_obj_loss_pos += obj_loss.sum()

            #     # 使用 FocalLoss 計算類別損失
            #     pred_cls_pos = pred_cls[obj_mask]
            #     target_cls_pos = target_cls[obj_mask]
            #     cls_loss = self.focal_loss(pred_cls_pos, target_cls_pos)
            #     total_cls_loss += cls_loss.sum()

            # if num_neg > 0:
            #     # 使用 FocalLoss 計算 objectness 損失（負樣本）
            #     pred_conf_neg = pred_conf[noobj_mask]
            #     target_conf_neg = torch.zeros_like(pred_conf_neg)
            #     noobj_loss = self.focal_loss(pred_conf_neg, target_conf_neg)
            #     total_obj_loss_neg += noobj_loss.sum()
            pred_boxes = pred[..., 0:4]
            pred_conf = pred[..., 4]
            pred_cls = pred[..., 5:]

            target_boxes = gt[..., 0:4]
            target_conf = gt[..., 4]
            target_cls = gt[..., 5:]

            # 正負樣本 mask
            obj_mask = target_conf > 0
            noobj_mask = target_conf == 0

            num_pos = obj_mask.sum().item()
            num_neg = noobj_mask.sum().item()
            total_num_pos += num_pos
            total_num_neg += num_neg

            # 1. box loss：只累加正樣本的 GIoU loss
            if num_pos > 0:
                giou_loss_map = self.box_loss(pred_boxes, target_boxes, anchors)  # [B,S,S,A]
                total_box_loss += giou_loss_map[obj_mask].sum()

            # 2. 正樣本 objectness：target = IoU (IoU-aware confidence)
            if num_pos > 0:
                pos_indices = obj_mask.nonzero(as_tuple=False)   # [N,4]: (b, y, x, a)
                pred_boxes_pos = pred_boxes[obj_mask]
                target_boxes_pos = target_boxes[obj_mask]

                # 取對應 anchor
                anchor_tensor = torch.tensor(anchors, device=device, dtype=pred_boxes.dtype)
                anchor_indices = pos_indices[:, 3]
                anchors_for_pos = anchor_tensor[anchor_indices]

                # 取得該格座標
                gy = pos_indices[:, 1].float().unsqueeze(1)
                gx = pos_indices[:, 2].float().unsqueeze(1)

                eps = 1e-9

                # decode 預測中心
                pred_xy = torch.sigmoid(pred_boxes_pos[..., 0:2])       # cell 內偏移
                pred_xy_norm = (pred_xy + torch.cat([gx, gy], dim=1)) / grid

                # decode GT 中心（target_boxes[...,0:2] 為 cell 內偏移）
                target_xy = target_boxes_pos[..., 0:2]
                target_xy_norm = (target_xy + torch.cat([gx, gy], dim=1)) / grid

                # 寬高（pred 用 anchor*exp，gt 直接是 normalized w,h）
                pred_wh = torch.exp(pred_boxes_pos[..., 2:4]) * anchors_for_pos
                target_wh = target_boxes_pos[..., 2:4]

                # 轉角點
                pred_x1y1 = pred_xy_norm - pred_wh / 2.0
                pred_x2y2 = pred_xy_norm + pred_wh / 2.0
                tgt_x1y1 = target_xy_norm - target_wh / 2.0
                tgt_x2y2 = target_xy_norm + target_wh / 2.0

                # 交集
                inter_x1y1 = torch.max(pred_x1y1, tgt_x1y1)
                inter_x2y2 = torch.min(pred_x2y2, tgt_x2y2)
                inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                # 並集
                pred_area = (pred_wh[:, 0] * pred_wh[:, 1]).clamp(min=0)
                tgt_area = (target_wh[:, 0] * target_wh[:, 1]).clamp(min=0)
                union = pred_area + tgt_area - inter_area + eps
                iou = (inter_area / union).clamp(0.0, 1.0).detach()

                # 正樣本 objectness Focal loss
                pred_conf_pos = pred_conf[obj_mask]
                obj_loss_pos = self.focal_loss(pred_conf_pos, iou)
                total_obj_loss_pos += obj_loss_pos.sum()

                # 3. 分類損失：只對正樣本
                pred_cls_pos = pred_cls[obj_mask]
                target_cls_pos = target_cls[obj_mask]
                cls_loss = self.focal_loss(pred_cls_pos, target_cls_pos)
                total_cls_loss += cls_loss.sum()
            # 若沒有正樣本，保持梯度圖連續
            else:
                total_box_loss += pred_boxes[obj_mask].sum() * 0.0
                total_obj_loss_pos += pred_conf[obj_mask].sum() * 0.0
                total_cls_loss += pred_cls[obj_mask].sum() * 0.0

            # 4. 負樣本 objectness：target = 0
            if num_neg > 0:
                pred_conf_neg = pred_conf[noobj_mask]
                target_conf_neg = torch.zeros_like(pred_conf_neg)
                noobj_loss = self.focal_loss(pred_conf_neg, target_conf_neg)
                total_obj_loss_neg += noobj_loss.sum()
            else:
                total_obj_loss_neg += pred_conf[noobj_mask].sum() * 0.0
            ##########################################################
        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)

        total_box_loss = total_box_loss / pos_denom
        total_obj_loss = total_obj_loss_pos / pos_denom
        total_cls_loss = total_cls_loss / pos_denom
        total_noobj_loss = total_obj_loss_neg / neg_denom

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

