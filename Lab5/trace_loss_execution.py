"""
逐行執行 YOLOv3Loss 計算並記錄每個變數的真實值
輸出到 loss_execution_trace.txt
"""
import torch
import numpy as np
from yolo_loss import YOLOv3Loss
import sys
from datetime import datetime

class ExecutionTracer:
    """記錄執行過程的所有變數值"""
    def __init__(self, output_file):
        self.output_file = output_file
        self.file = open(output_file, 'w', encoding='utf-8')
        self.indent_level = 0
        
    def write(self, text):
        """寫入帶縮排的文字"""
        indent = "  " * self.indent_level
        self.file.write(indent + text + "\n")
        self.file.flush()
        
    def write_separator(self, char="=", length=80):
        """寫入分隔線"""
        self.file.write(char * length + "\n")
        self.file.flush()
        
    def write_header(self, title):
        """寫入標題"""
        self.write_separator()
        self.write(title)
        self.write_separator()
        
    def write_variable(self, name, value, show_stats=True, show_values=False, max_show=10):
        """記錄變數資訊"""
        self.write(f"\n{name}:")
        self.indent_level += 1
        
        if isinstance(value, torch.Tensor):
            self.write(f"Type: Tensor")
            self.write(f"Shape: {list(value.shape)}")
            self.write(f"Dtype: {value.dtype}")
            self.write(f"Device: {value.device}")
            
            if value.numel() > 0 and show_stats:
                # 只對浮點數計算統計值
                if value.dtype in [torch.float16, torch.float32, torch.float64]:
                    self.write(f"Min: {value.min().item():.8f}")
                    self.write(f"Max: {value.max().item():.8f}")
                    self.write(f"Mean: {value.mean().item():.8f}")
                    self.write(f"Std: {value.std().item():.8f}")
                else:
                    self.write(f"Min: {value.min().item()}")
                    self.write(f"Max: {value.max().item()}")
                
            if show_values:
                if value.numel() <= max_show:
                    self.write(f"Values: {value.tolist()}")
                else:
                    flat = value.flatten()
                    self.write(f"First {max_show} values: {flat[:max_show].tolist()}")
                    
        elif isinstance(value, (int, float, np.number)):
            self.write(f"Type: {type(value).__name__}")
            self.write(f"Value: {value}")
            
        elif isinstance(value, (list, tuple)):
            self.write(f"Type: {type(value).__name__}")
            self.write(f"Length: {len(value)}")
            if len(value) <= max_show:
                self.write(f"Values: {value}")
            else:
                self.write(f"First {max_show} values: {value[:max_show]}")
        else:
            self.write(f"Type: {type(value).__name__}")
            self.write(f"Value: {value}")
            
        self.indent_level -= 1
        
    def close(self):
        """關閉文件"""
        self.file.close()


def create_simple_data():
    """創建測試數據"""
    batch_size = 1
    grid = 13
    num_anchors = 3
    num_classes = 20
    
    pred = torch.randn(batch_size, grid, grid, num_anchors * (5 + num_classes))
    target = torch.zeros(batch_size, grid, grid, num_anchors, 5 + num_classes)
    
    # 物體 1
    target[0, 6, 6, 0, 0] = 0.5
    target[0, 6, 6, 0, 1] = 0.5
    target[0, 6, 6, 0, 2] = 0.3
    target[0, 6, 6, 0, 3] = 0.4
    target[0, 6, 6, 0, 4] = 1.0
    target[0, 6, 6, 0, 5 + 5] = 1.0
    
    # 物體 2
    target[0, 8, 10, 1, 0] = 0.7
    target[0, 8, 10, 1, 1] = 0.3
    target[0, 8, 10, 1, 2] = 0.5
    target[0, 8, 10, 1, 3] = 0.6
    target[0, 8, 10, 1, 4] = 1.0
    target[0, 8, 10, 1, 5 + 12] = 1.0
    
    return [pred], [target]


def trace_loss_calculation():
    """逐行追蹤損失計算"""
    
    tracer = ExecutionTracer('loss_execution_trace.txt')
    
    tracer.write_header(f"YOLOv3 Loss 執行追蹤 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化
    tracer.write("\n" + "="*80)
    tracer.write("初始化階段")
    tracer.write("="*80)
    
    anchors = [[(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)]]
    tracer.write_variable("anchors", anchors, show_stats=False)
    
    criterion = YOLOv3Loss(
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
        lambda_class=1.0,
        anchors=anchors
    )
    
    tracer.write_variable("lambda_coord", criterion.lambda_coord, show_stats=False)
    tracer.write_variable("lambda_obj", criterion.lambda_obj, show_stats=False)
    tracer.write_variable("lambda_noobj", criterion.lambda_noobj, show_stats=False)
    tracer.write_variable("lambda_class", criterion.lambda_class, show_stats=False)
    tracer.write_variable("focal_loss.alpha", criterion.focal_loss.alpha, show_stats=False)
    tracer.write_variable("focal_loss.gamma", criterion.focal_loss.gamma, show_stats=False)
    tracer.write_variable("box_loss.type", criterion.box_loss.type, show_stats=False)
    
    # 創建數據
    tracer.write("\n" + "="*80)
    tracer.write("創建測試數據")
    tracer.write("="*80)
    
    predictions, targets = create_simple_data()
    pred = predictions[0]
    gt = targets[0]
    
    tracer.write_variable("pred (原始)", pred, show_stats=True, show_values=False)
    tracer.write_variable("gt (原始)", gt, show_stats=True, show_values=False)
    
    # 開始前向傳播
    tracer.write("\n" + "="*80)
    tracer.write("前向傳播 - 開始")
    tracer.write("="*80)
    
    device = pred.device
    tracer.write_variable("device", str(device), show_stats=False)
    
    # 重塑
    tracer.write("\n>>> 執行: bsz, grid, _, num_anchors_total = pred.shape")
    bsz, grid, _, num_anchors_total = pred.shape
    tracer.write_variable("bsz", bsz, show_stats=False)
    tracer.write_variable("grid", grid, show_stats=False)
    tracer.write_variable("num_anchors_total", num_anchors_total, show_stats=False)
    
    tracer.write("\n>>> 執行: num_anchors = len(anchors[0])")
    num_anchors = len(anchors[0])
    tracer.write_variable("num_anchors", num_anchors, show_stats=False)
    
    tracer.write("\n>>> 執行: pred_reshaped = pred.view(bsz, grid, grid, num_anchors, -1)")
    pred_reshaped = pred.view(bsz, grid, grid, num_anchors, -1)
    tracer.write_variable("pred_reshaped", pred_reshaped, show_stats=True, show_values=False)
    
    # 分離各部分
    tracer.write("\n>>> 執行: pred_boxes = pred_reshaped[..., 0:4]")
    pred_boxes = pred_reshaped[..., 0:4]
    tracer.write_variable("pred_boxes", pred_boxes, show_stats=True, show_values=False)
    
    tracer.write("\n>>> 執行: pred_conf = pred_reshaped[..., 4]")
    pred_conf = pred_reshaped[..., 4]
    tracer.write_variable("pred_conf", pred_conf, show_stats=True, show_values=False)
    
    tracer.write("\n>>> 執行: pred_cls = pred_reshaped[..., 5:]")
    pred_cls = pred_reshaped[..., 5:]
    tracer.write_variable("pred_cls", pred_cls, show_stats=True, show_values=False)
    
    tracer.write("\n>>> 執行: target_boxes = gt[..., 0:4]")
    target_boxes = gt[..., 0:4]
    tracer.write_variable("target_boxes", target_boxes, show_stats=True, show_values=False)
    
    tracer.write("\n>>> 執行: target_conf = gt[..., 4]")
    target_conf = gt[..., 4]
    tracer.write_variable("target_conf", target_conf, show_stats=True, show_values=False)
    
    tracer.write("\n>>> 執行: target_cls = gt[..., 5:]")
    target_cls = gt[..., 5:]
    tracer.write_variable("target_cls", target_cls, show_stats=True, show_values=False)
    
    # 創建遮罩
    tracer.write("\n>>> 執行: obj_mask = gt[..., 4] > 0")
    obj_mask = gt[..., 4] > 0
    tracer.write_variable("obj_mask", obj_mask, show_stats=False, show_values=False)
    
    tracer.write("\n>>> 執行: noobj_mask = gt[..., 4] == 0")
    noobj_mask = gt[..., 4] == 0
    tracer.write_variable("noobj_mask", noobj_mask, show_stats=False, show_values=False)
    
    tracer.write("\n>>> 執行: num_pos = obj_mask.sum().item()")
    num_pos = obj_mask.sum().item()
    tracer.write_variable("num_pos", num_pos, show_stats=False)
    
    tracer.write("\n>>> 執行: num_neg = noobj_mask.sum().item()")
    num_neg = noobj_mask.sum().item()
    tracer.write_variable("num_neg", num_neg, show_stats=False)
    
    # 正樣本位置
    tracer.write("\n>>> 執行: pos_indices = obj_mask.nonzero(as_tuple=False)")
    pos_indices = obj_mask.nonzero(as_tuple=False)
    tracer.write_variable("pos_indices", pos_indices, show_stats=False, show_values=True)
    
    # 顯示每個正樣本的詳細資訊
    tracer.write("\n" + "="*80)
    tracer.write("正樣本詳細資訊")
    tracer.write("="*80)
    
    for idx_num, idx in enumerate(pos_indices):
        b, y, x, a = idx.tolist()
        tracer.write(f"\n--- 正樣本 {idx_num + 1} ---")
        tracer.write(f"位置: batch={b}, grid_y={y}, grid_x={x}, anchor={a}")
        
        tracer.write("\n標籤資訊:")
        tracer.indent_level += 1
        tracer.write(f"x_offset: {target_boxes[b,y,x,a,0].item():.6f}")
        tracer.write(f"y_offset: {target_boxes[b,y,x,a,1].item():.6f}")
        tracer.write(f"w: {target_boxes[b,y,x,a,2].item():.6f}")
        tracer.write(f"h: {target_boxes[b,y,x,a,3].item():.6f}")
        tracer.write(f"objectness: {target_conf[b,y,x,a].item():.6f}")
        cls_idx = target_cls[b,y,x,a].argmax().item()
        tracer.write(f"class: {cls_idx}")
        tracer.indent_level -= 1
        
        tracer.write("\n預測資訊 (raw logits):")
        tracer.indent_level += 1
        tracer.write(f"x_raw: {pred_boxes[b,y,x,a,0].item():.6f}")
        tracer.write(f"y_raw: {pred_boxes[b,y,x,a,1].item():.6f}")
        tracer.write(f"w_raw: {pred_boxes[b,y,x,a,2].item():.6f}")
        tracer.write(f"h_raw: {pred_boxes[b,y,x,a,3].item():.6f}")
        tracer.write(f"conf_raw: {pred_conf[b,y,x,a].item():.6f}")
        tracer.indent_level -= 1
    
    # Box Loss 計算
    if num_pos > 0:
        tracer.write("\n" + "="*80)
        tracer.write("計算 Box Loss (使用 BoxLoss 類)")
        tracer.write("="*80)
        
        tracer.write("\n>>> 執行: box_loss_all = criterion.box_loss(pred_boxes, target_boxes, anchors[0])")
        box_loss_all = criterion.box_loss(pred_boxes, target_boxes, anchors[0])
        tracer.write_variable("box_loss_all (所有位置)", box_loss_all, show_stats=True, show_values=False)
        
        tracer.write("\n>>> 執行: box_loss_pos = box_loss_all[obj_mask]")
        box_loss_pos = box_loss_all[obj_mask]
        tracer.write_variable("box_loss_pos (正樣本)", box_loss_pos, show_stats=True, show_values=True)
        
        tracer.write("\n>>> 執行: total_box_loss = box_loss_pos.sum()")
        total_box_loss = box_loss_pos.sum()
        tracer.write_variable("total_box_loss (總和)", total_box_loss.item(), show_stats=False)
        
        # 計算 IoU
        tracer.write("\n" + "="*80)
        tracer.write("計算 IoU (用於 objectness target)")
        tracer.write("="*80)
        
        with torch.no_grad():
            tracer.write("\n>>> 執行: pred_boxes_pos = pred_boxes[obj_mask]")
            pred_boxes_pos = pred_boxes[obj_mask]
            tracer.write_variable("pred_boxes_pos", pred_boxes_pos, show_values=True)
            
            tracer.write("\n>>> 執行: target_boxes_pos = target_boxes[obj_mask]")
            target_boxes_pos = target_boxes[obj_mask]
            tracer.write_variable("target_boxes_pos", target_boxes_pos, show_values=True)
            
            tracer.write("\n>>> 執行: anchor_indices = pos_indices[:, 3]")
            anchor_indices = pos_indices[:, 3]
            tracer.write_variable("anchor_indices", anchor_indices, show_values=True)
            
            tracer.write("\n>>> 執行: anchor_tensor = torch.tensor(anchors[0])")
            anchor_tensor = torch.tensor(anchors[0], dtype=pred_boxes.dtype)
            tracer.write_variable("anchor_tensor", anchor_tensor, show_values=True)
            
            tracer.write("\n>>> 執行: anchors_for_pos = anchor_tensor[anchor_indices]")
            anchors_for_pos = anchor_tensor[anchor_indices]
            tracer.write_variable("anchors_for_pos", anchors_for_pos, show_values=True)
            
            tracer.write("\n>>> 執行: pred_xy = torch.sigmoid(pred_boxes_pos[:, 0:2])")
            pred_xy = torch.sigmoid(pred_boxes_pos[:, 0:2])
            tracer.write_variable("pred_xy", pred_xy, show_values=True)
            
            tracer.write("\n>>> 執行: pred_wh = torch.exp(pred_boxes_pos[:, 2:4]) * anchors_for_pos")
            pred_wh = torch.exp(pred_boxes_pos[:, 2:4]) * anchors_for_pos
            tracer.write_variable("pred_wh", pred_wh, show_values=True)
            
            tracer.write("\n>>> 執行: target_xy = target_boxes_pos[:, 0:2]")
            target_xy = target_boxes_pos[:, 0:2]
            tracer.write_variable("target_xy", target_xy, show_values=True)
            
            tracer.write("\n>>> 執行: target_wh = target_boxes_pos[:, 2:4]")
            target_wh = target_boxes_pos[:, 2:4]
            tracer.write_variable("target_wh", target_wh, show_values=True)
            
            tracer.write("\n>>> 執行: grid_coords = pos_indices[:, 1:3].float()")
            grid_coords = pos_indices[:, 1:3].float()
            tracer.write_variable("grid_coords", grid_coords, show_values=True)
            
            tracer.write("\n>>> 執行: grid_x = grid_coords[:, 1:2]")
            grid_x = grid_coords[:, 1:2]
            tracer.write_variable("grid_x", grid_x, show_values=True)
            
            tracer.write("\n>>> 執行: grid_y = grid_coords[:, 0:1]")
            grid_y = grid_coords[:, 0:1]
            tracer.write_variable("grid_y", grid_y, show_values=True)
            
            tracer.write("\n>>> 執行: pred_xy_norm = (pred_xy + torch.cat([grid_x, grid_y], dim=1)) / grid")
            pred_xy_norm = (pred_xy + torch.cat([grid_x, grid_y], dim=1)) / grid
            tracer.write_variable("pred_xy_norm", pred_xy_norm, show_values=True)
            
            tracer.write("\n>>> 執行: target_xy_norm = (target_xy + torch.cat([grid_x, grid_y], dim=1)) / grid")
            target_xy_norm = (target_xy + torch.cat([grid_x, grid_y], dim=1)) / grid
            tracer.write_variable("target_xy_norm", target_xy_norm, show_values=True)
            
            tracer.write("\n>>> 執行: pred_x1y1 = pred_xy_norm - pred_wh / 2")
            pred_x1y1 = pred_xy_norm - pred_wh / 2
            tracer.write_variable("pred_x1y1", pred_x1y1, show_values=True)
            
            tracer.write("\n>>> 執行: pred_x2y2 = pred_xy_norm + pred_wh / 2")
            pred_x2y2 = pred_xy_norm + pred_wh / 2
            tracer.write_variable("pred_x2y2", pred_x2y2, show_values=True)
            
            tracer.write("\n>>> 執行: target_x1y1 = target_xy_norm - target_wh / 2")
            target_x1y1 = target_xy_norm - target_wh / 2
            tracer.write_variable("target_x1y1", target_x1y1, show_values=True)
            
            tracer.write("\n>>> 執行: target_x2y2 = target_xy_norm + target_wh / 2")
            target_x2y2 = target_xy_norm + target_wh / 2
            tracer.write_variable("target_x2y2", target_x2y2, show_values=True)
            
            tracer.write("\n>>> 執行: inter_x1y1 = torch.max(pred_x1y1, target_x1y1)")
            inter_x1y1 = torch.max(pred_x1y1, target_x1y1)
            tracer.write_variable("inter_x1y1", inter_x1y1, show_values=True)
            
            tracer.write("\n>>> 執行: inter_x2y2 = torch.min(pred_x2y2, target_x2y2)")
            inter_x2y2 = torch.min(pred_x2y2, target_x2y2)
            tracer.write_variable("inter_x2y2", inter_x2y2, show_values=True)
            
            tracer.write("\n>>> 執行: inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)")
            inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
            tracer.write_variable("inter_wh", inter_wh, show_values=True)
            
            tracer.write("\n>>> 執行: inter_area = inter_wh[:, 0] * inter_wh[:, 1]")
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]
            tracer.write_variable("inter_area", inter_area, show_values=True)
            
            tracer.write("\n>>> 執行: pred_area = pred_wh[:, 0] * pred_wh[:, 1]")
            pred_area = pred_wh[:, 0] * pred_wh[:, 1]
            tracer.write_variable("pred_area", pred_area, show_values=True)
            
            tracer.write("\n>>> 執行: target_area = target_wh[:, 0] * target_wh[:, 1]")
            target_area = target_wh[:, 0] * target_wh[:, 1]
            tracer.write_variable("target_area", target_area, show_values=True)
            
            tracer.write("\n>>> 執行: union = pred_area + target_area - inter_area + 1e-9")
            union = pred_area + target_area - inter_area + 1e-9
            tracer.write_variable("union", union, show_values=True)
            
            tracer.write("\n>>> 執行: iou = inter_area / union")
            iou = inter_area / union
            tracer.write_variable("iou", iou, show_values=True)
            
            tracer.write("\n>>> 執行: ious = iou.clamp(0, 1)")
            ious = iou.clamp(0, 1)
            tracer.write_variable("ious", ious, show_values=True)
        
        # Objectness Loss
        tracer.write("\n" + "="*80)
        tracer.write("計算 Objectness Loss (使用 FocalLoss)")
        tracer.write("="*80)
        
        tracer.write("\n>>> 執行: pred_conf_pos = pred_conf[obj_mask]")
        pred_conf_pos = pred_conf[obj_mask]
        tracer.write_variable("pred_conf_pos (logits)", pred_conf_pos, show_values=True)
        
        tracer.write("\n>>> 執行: obj_loss = criterion.focal_loss(pred_conf_pos, ious)")
        obj_loss = criterion.focal_loss(pred_conf_pos, ious)
        tracer.write_variable("obj_loss", obj_loss, show_values=True)
        
        tracer.write("\n>>> 執行: total_obj_loss_pos = obj_loss.sum()")
        total_obj_loss_pos = obj_loss.sum()
        tracer.write_variable("total_obj_loss_pos", total_obj_loss_pos.item(), show_stats=False)
        
        # Class Loss
        tracer.write("\n" + "="*80)
        tracer.write("計算 Class Loss (使用 FocalLoss)")
        tracer.write("="*80)
        
        tracer.write("\n>>> 執行: pred_cls_pos = pred_cls[obj_mask]")
        pred_cls_pos = pred_cls[obj_mask]
        tracer.write_variable("pred_cls_pos", pred_cls_pos, show_stats=True, show_values=False)
        
        tracer.write("\n>>> 執行: target_cls_pos = target_cls[obj_mask]")
        target_cls_pos = target_cls[obj_mask]
        tracer.write_variable("target_cls_pos", target_cls_pos, show_stats=True, show_values=False)
        
        # 顯示每個樣本的類別資訊
        for i in range(len(pred_cls_pos)):
            tracer.write(f"\n正樣本 {i+1} 的類別資訊:")
            tracer.indent_level += 1
            target_class = target_cls_pos[i].argmax().item()
            tracer.write(f"標籤類別: {target_class}")
            tracer.write(f"該類別的預測 logit: {pred_cls_pos[i, target_class].item():.6f}")
            tracer.write(f"所有類別預測 logits 統計:")
            tracer.indent_level += 1
            tracer.write(f"min: {pred_cls_pos[i].min().item():.6f}")
            tracer.write(f"max: {pred_cls_pos[i].max().item():.6f}")
            tracer.write(f"mean: {pred_cls_pos[i].mean().item():.6f}")
            tracer.indent_level -= 2
        
        tracer.write("\n>>> 執行: cls_loss = criterion.focal_loss(pred_cls_pos, target_cls_pos)")
        cls_loss = criterion.focal_loss(pred_cls_pos, target_cls_pos)
        tracer.write_variable("cls_loss", cls_loss, show_stats=True, show_values=False)
        
        tracer.write("\n>>> 執行: total_cls_loss = cls_loss.sum()")
        total_cls_loss = cls_loss.sum()
        tracer.write_variable("total_cls_loss", total_cls_loss.item(), show_stats=False)
    
    # No-Objectness Loss
    if num_neg > 0:
        tracer.write("\n" + "="*80)
        tracer.write("計算 No-Objectness Loss (負樣本)")
        tracer.write("="*80)
        
        tracer.write("\n>>> 執行: pred_conf_neg = pred_conf[noobj_mask]")
        pred_conf_neg = pred_conf[noobj_mask]
        tracer.write_variable("pred_conf_neg", pred_conf_neg, show_stats=True, show_values=False)
        
        tracer.write("\n>>> 執行: target_conf_neg = torch.zeros_like(pred_conf_neg)")
        target_conf_neg = torch.zeros_like(pred_conf_neg)
        tracer.write_variable("target_conf_neg", target_conf_neg, show_stats=True, show_values=False)
        
        tracer.write("\n>>> 執行: noobj_loss = criterion.focal_loss(pred_conf_neg, target_conf_neg)")
        noobj_loss = criterion.focal_loss(pred_conf_neg, target_conf_neg)
        tracer.write_variable("noobj_loss", noobj_loss, show_stats=True, show_values=False)
        
        tracer.write("\n>>> 執行: total_obj_loss_neg = noobj_loss.sum()")
        total_obj_loss_neg = noobj_loss.sum()
        tracer.write_variable("total_obj_loss_neg", total_obj_loss_neg.item(), show_stats=False)
    
    # 歸一化和加權
    tracer.write("\n" + "="*80)
    tracer.write("歸一化和加權")
    tracer.write("="*80)
    
    tracer.write("\n>>> 執行: pos_denom = max(num_pos, 1)")
    pos_denom = max(num_pos, 1)
    tracer.write_variable("pos_denom", pos_denom, show_stats=False)
    
    tracer.write("\n>>> 執行: neg_denom = max(num_neg, 1)")
    neg_denom = max(num_neg, 1)
    tracer.write_variable("neg_denom", neg_denom, show_stats=False)
    
    if num_pos > 0:
        tracer.write("\n>>> 執行: total_box_loss = total_box_loss / pos_denom")
        total_box_loss = total_box_loss / pos_denom
        tracer.write_variable("total_box_loss (歸一化)", total_box_loss.item(), show_stats=False)
        
        tracer.write("\n>>> 執行: total_obj_loss = total_obj_loss_pos / pos_denom")
        total_obj_loss = total_obj_loss_pos / pos_denom
        tracer.write_variable("total_obj_loss (歸一化)", total_obj_loss.item(), show_stats=False)
        
        tracer.write("\n>>> 執行: total_cls_loss = total_cls_loss / pos_denom")
        total_cls_loss = total_cls_loss / pos_denom
        tracer.write_variable("total_cls_loss (歸一化)", total_cls_loss.item(), show_stats=False)
    
    if num_neg > 0:
        tracer.write("\n>>> 執行: total_noobj_loss = total_obj_loss_neg / neg_denom")
        total_noobj_loss = total_obj_loss_neg / neg_denom
        tracer.write_variable("total_noobj_loss (歸一化)", total_noobj_loss.item(), show_stats=False)
    
    # 計算總損失
    tracer.write("\n" + "="*80)
    tracer.write("計算總損失")
    tracer.write("="*80)
    
    tracer.write("\n>>> 執行: total_loss = lambda_coord * total_box_loss + ...")
    total_loss = (
        criterion.lambda_coord * total_box_loss +
        criterion.lambda_obj * total_obj_loss +
        criterion.lambda_noobj * total_noobj_loss +
        criterion.lambda_class * total_cls_loss
    )
    tracer.write_variable("total_loss", total_loss.item(), show_stats=False)
    
    # 詳細展開
    tracer.write("\n加權計算詳細:")
    tracer.indent_level += 1
    box_weighted = (criterion.lambda_coord * total_box_loss).item()
    obj_weighted = (criterion.lambda_obj * total_obj_loss).item()
    noobj_weighted = (criterion.lambda_noobj * total_noobj_loss).item()
    cls_weighted = (criterion.lambda_class * total_cls_loss).item()
    
    tracer.write(f"Box Loss:   {total_box_loss.item():.8f} × {criterion.lambda_coord} = {box_weighted:.8f}")
    tracer.write(f"Obj Loss:   {total_obj_loss.item():.8f} × {criterion.lambda_obj} = {obj_weighted:.8f}")
    tracer.write(f"NoObj Loss: {total_noobj_loss.item():.8f} × {criterion.lambda_noobj} = {noobj_weighted:.8f}")
    tracer.write(f"Class Loss: {total_cls_loss.item():.8f} × {criterion.lambda_class} = {cls_weighted:.8f}")
    tracer.write(f"---")
    tracer.write(f"Total:      {box_weighted + obj_weighted + noobj_weighted + cls_weighted:.8f}")
    tracer.indent_level -= 1
    
    # 驗證
    tracer.write("\n" + "="*80)
    tracer.write("使用完整損失函數驗證")
    tracer.write("="*80)
    
    loss_dict = criterion(predictions, targets)
    
    tracer.write("\n>>> 執行: loss_dict = criterion(predictions, targets)")
    tracer.write_variable("loss_dict['total']", loss_dict['total'].item(), show_stats=False)
    tracer.write_variable("loss_dict['box']", loss_dict['box'].item(), show_stats=False)
    tracer.write_variable("loss_dict['obj']", loss_dict['obj'].item(), show_stats=False)
    tracer.write_variable("loss_dict['noobj']", loss_dict['noobj'].item(), show_stats=False)
    tracer.write_variable("loss_dict['cls']", loss_dict['cls'].item(), show_stats=False)
    
    tracer.write("\n" + "="*80)
    tracer.write("執行追蹤完成")
    tracer.write("="*80)
    
    tracer.close()
    
    print(f"✓ 執行追蹤已保存到: loss_execution_trace.txt")
    print(f"總行數: {sum(1 for _ in open('loss_execution_trace.txt', encoding='utf-8'))}")


if __name__ == "__main__":
    trace_loss_calculation()
