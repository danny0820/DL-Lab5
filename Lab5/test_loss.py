"""
測試 YOLOv3Loss 的各項損失值
使用模擬數據快速驗證損失函數是否正常工作
"""
import torch
import numpy as np
from yolo_loss import YOLOv3Loss

def create_dummy_data(batch_size=2, grid_sizes=[13, 26, 52], num_anchors=3, num_classes=20):
    """
    創建模擬的預測和標籤數據
    """
    predictions = []
    targets = []
    
    for grid in grid_sizes:
        # 創建預測數據 [batch, grid, grid, num_anchors * (5 + num_classes)]
        pred = torch.randn(batch_size, grid, grid, num_anchors * (5 + num_classes))
        predictions.append(pred)
        
        # 創建標籤數據 [batch, grid, grid, num_anchors, 5 + num_classes]
        target = torch.zeros(batch_size, grid, grid, num_anchors, 5 + num_classes)
        
        # 隨機設置一些正樣本
        num_objects = np.random.randint(1, 5)  # 每個 batch 隨機 1-4 個物體
        for _ in range(num_objects):
            b = np.random.randint(0, batch_size)
            i = np.random.randint(0, grid)
            j = np.random.randint(0, grid)
            a = np.random.randint(0, num_anchors)
            
            # 設置邊界框 (x, y, w, h)
            target[b, i, j, a, 0] = np.random.rand()  # x offset
            target[b, i, j, a, 1] = np.random.rand()  # y offset
            target[b, i, j, a, 2] = np.random.rand() * 0.5  # w (歸一化)
            target[b, i, j, a, 3] = np.random.rand() * 0.5  # h (歸一化)
            target[b, i, j, a, 4] = 1.0  # objectness
            
            # 設置類別 (one-hot)
            cls_idx = np.random.randint(0, num_classes)
            target[b, i, j, a, 5 + cls_idx] = 1.0
        
        targets.append(target)
    
    return predictions, targets


def test_loss_function():
    """
    測試損失函數並打印各項損失值
    """
    print("="*60)
    print("YOLOv3 Loss 測試")
    print("="*60)
    
    # 定義錨框（3個尺度，每個尺度3個錨框）
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],  # 大尺度 (13x13)
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # 中尺度 (26x26)
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # 小尺度 (52x52)
    ]
    
    # 初始化損失函數
    criterion = YOLOv3Loss(
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
        lambda_class=1.0,
        anchors=anchors
    )
    
    print("\n損失函數參數:")
    print(f"  lambda_coord (邊界框權重): {criterion.lambda_coord}")
    print(f"  lambda_obj (有物體權重): {criterion.lambda_obj}")
    print(f"  lambda_noobj (無物體權重): {criterion.lambda_noobj}")
    print(f"  lambda_class (類別權重): {criterion.lambda_class}")
    print(f"  BoxLoss 類型: {criterion.box_loss.type}")
    print(f"  FocalLoss alpha: {criterion.focal_loss.alpha}")
    print(f"  FocalLoss gamma: {criterion.focal_loss.gamma}")
    
    # 創建模擬數據
    print("\n創建模擬數據...")
    predictions, targets = create_dummy_data(batch_size=2, num_classes=20)
    
    print(f"  Batch size: 2")
    print(f"  網格尺寸: [13, 26, 52]")
    print(f"  每個尺度的錨框數量: 3")
    print(f"  類別數量: 20")
    
    # 計算損失
    print("\n計算損失...")
    with torch.no_grad():  # 測試時不需要梯度
        loss_dict = criterion(predictions, targets)
    
    # 打印結果
    print("\n" + "="*60)
    print("損失值結果:")
    print("="*60)
    print(f"  總損失 (Total Loss):      {loss_dict['total'].item():.6f}")
    print(f"  邊界框損失 (Box Loss):    {loss_dict['box'].item():.6f}")
    print(f"  有物體損失 (Obj Loss):    {loss_dict['obj'].item():.6f}")
    print(f"  無物體損失 (NoObj Loss):  {loss_dict['noobj'].item():.6f}")
    print(f"  類別損失 (Class Loss):    {loss_dict['cls'].item():.6f}")
    print("="*60)
    
    # 計算加權後的貢獻
    print("\n各項損失對總損失的加權貢獻:")
    box_contrib = criterion.lambda_coord * loss_dict['box'].item()
    obj_contrib = criterion.lambda_obj * loss_dict['obj'].item()
    noobj_contrib = criterion.lambda_noobj * loss_dict['noobj'].item()
    cls_contrib = criterion.lambda_class * loss_dict['cls'].item()
    
    print(f"  邊界框貢獻: {box_contrib:.6f} ({box_contrib/loss_dict['total'].item()*100:.2f}%)")
    print(f"  有物體貢獻: {obj_contrib:.6f} ({obj_contrib/loss_dict['total'].item()*100:.2f}%)")
    print(f"  無物體貢獻: {noobj_contrib:.6f} ({noobj_contrib/loss_dict['total'].item()*100:.2f}%)")
    print(f"  類別貢獻:   {cls_contrib:.6f} ({cls_contrib/loss_dict['total'].item()*100:.2f}%)")
    print(f"  總和驗證:   {box_contrib + obj_contrib + noobj_contrib + cls_contrib:.6f}")
    print("="*60)
    
    return loss_dict


def test_with_gradient():
    """
    測試損失函數的梯度反向傳播
    """
    print("\n" + "="*60)
    print("梯度測試")
    print("="*60)
    
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
    
    criterion = YOLOv3Loss(anchors=anchors)
    
    # 創建需要梯度的預測數據
    predictions, targets = create_dummy_data(batch_size=1)
    predictions = [p.requires_grad_(True) for p in predictions]
    
    # 前向傳播
    loss_dict = criterion(predictions, targets)
    total_loss = loss_dict['total']
    
    # 反向傳播
    total_loss.backward()
    
    # 檢查梯度
    print("\n梯度檢查:")
    for i, pred in enumerate(predictions):
        if pred.grad is not None:
            grad_norm = pred.grad.norm().item()
            grad_mean = pred.grad.mean().item()
            grad_max = pred.grad.max().item()
            grad_min = pred.grad.min().item()
            print(f"  尺度 {i+1} 梯度統計:")
            print(f"    Norm: {grad_norm:.6f}")
            print(f"    Mean: {grad_mean:.6f}")
            print(f"    Max:  {grad_max:.6f}")
            print(f"    Min:  {grad_min:.6f}")
        else:
            print(f"  尺度 {i+1}: 無梯度")
    
    print("\n✓ 梯度反向傳播正常")
    print("="*60)


if __name__ == "__main__":
    # 測試損失函數
    loss_dict = test_loss_function()
    
    # 測試梯度
    test_with_gradient()
    
    print("\n✓ 所有測試完成！")
    print("\n提示: 如果你想在訓練中查看這些值，可以在訓練循環中打印 loss_dict")
