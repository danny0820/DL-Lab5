# YOLOv3 模型評估與Loss解釋指南

## 📊 目錄
1. [如何知道預測效果](#如何知道預測效果)
2. [各項Loss的意義](#各項loss的意義)
3. [評估指標解釋](#評估指標解釋)
4. [如何判斷模型好壞](#如何判斷模型好壞)

---

## 🎯 如何知道預測效果

### 方法 1：訓練時的 mAP 評估（已實現）

**train.py 每5個epoch自動評估：**

```python
# 程式碼位置：train.py Line 267-275
if (epoch + 1) % 5 == 0:
    print('\n' + '-'*60)
    print('📈 Evaluating mAP on validation set...')
    print('-'*60)
    val_aps = evaluate(model, eval_loader)
    mean_ap = np.mean(val_aps)
    print(f'🎯 Epoch {epoch+1} - mAP: {mean_ap:.4f}')
    print('-'*60)
```

**輸出示例：**
```
------------------------------------------------------------
📈 Evaluating mAP on validation set...
------------------------------------------------------------
🎯 Epoch 5 - mAP: 0.4521
------------------------------------------------------------
```

**如何解讀：**
- mAP = 0.45：表示模型平均準確率為45%
- mAP範圍：0.0 ~ 1.0（越高越好）
- 目標：
  - mAP > 0.50：基本合格
  - mAP > 0.60：良好
  - mAP > 0.70：優秀
  - mAP > 0.80：非常優秀

---

### 方法 2：使用 predict_test.py 生成預測結果

**步驟：**

```bash
# 在訓練完成後運行預測
python predict_test.py
```

**predict_test.py 的功能：**
1. 載入最佳模型 `checkpoints/best_detector.pth`
2. 對測試集（8920張圖片）進行預測
3. 生成 `result.csv`（Kaggle提交格式）

**輸出格式（result.csv）：**
```csv
image_id,label_list
2007_000027,"14 0.315 0.229 0.445 0.478 0.972;..."
2007_000032,"11 0.156 0.129 0.712 0.893 0.965;..."
```

**解讀：**
- 每行是一張圖片的預測結果
- 格式：`class_id x1 y1 x2 y2 confidence;...`
- confidence：模型對該預測的信心（0~1）

---

### 方法 3：視覺化預測結果（需自行添加）

**創建可視化腳本（visualize_predictions.py）：**

```python
import torch
import cv2
import numpy as np
from src.yolo import getODmodel
from src.config import CLASSES

# 載入模型
model = getODmodel()
model.load_state_dict(torch.load('checkpoints/best_detector.pth'))
model.eval()

# 載入圖片
img = cv2.imread('test_image.jpg')

# 預測
with torch.no_grad():
    pred = model(preprocess(img))
    
# 繪製框框
for box in pred:
    class_id, x1, y1, x2, y2, conf = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{CLASSES[class_id]}: {conf:.2f}"
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('prediction.jpg', img)
```

**使用：**
```bash
python visualize_predictions.py
```

**結果：**
- 生成帶有預測框的圖片
- 可以直觀看到模型預測的物體位置和類別

---

### 方法 4：查看驗證集 Loss

**在訓練過程中觀察：**

```
📊 Validation Summary:
  Total Loss:  10.2345   ← 總損失（越低越好）
  Box Loss:    3.8920    ← 邊界框損失
  Obj Loss:    4.5670    ← 物體置信度損失
  Class Loss:  1.7755    ← 分類損失
```

**判斷標準：**
- **Loss 下降趨勢**：損失隨epoch減少 → 模型在學習
- **Loss 穩定**：損失不再下降 → 模型收斂
- **Loss 上升**：損失增加 → 過擬合或學習率問題

**對比訓練/驗證Loss：**
- 訓練Loss << 驗證Loss：**過擬合**（模型記憶訓練集）
- 訓練Loss ≈ 驗證Loss：**泛化良好**（模型學到通用特徵）

---

## 📚 各項Loss的意義

### 1. **Box Loss（邊界框損失）**

**作用：**
- 確保預測框的**位置**和**大小**與真實框接近

**計算方法：**
```python
# yolo_loss.py Line 179-191
# 1. 計算 GIoU（Generalized IoU）
iou_loss = compute_giou(pred_boxes, target_boxes)
box_loss = (1 - iou_loss).mean()
```

**包含兩部分：**

#### a) **GIoU Loss（位置損失）**

**公式：**
```
GIoU = IoU - |C - (A ∪ B)| / |C|
Loss = 1 - GIoU
```

**其中：**
- IoU：預測框與真實框的交集/聯集
- C：包含兩框的最小外接框
- A：預測框面積
- B：真實框面積

**意義：**
- IoU = 1.0：預測框完全重合真實框（完美預測）
- IoU = 0.5：預測框覆蓋50%真實框（基本合格）
- IoU = 0.0：預測框與真實框完全不重疊（預測失敗）

**為什麼用GIoU而非傳統IoU：**
1. ✅ 傳統IoU：兩框不重疊時梯度為0（無法學習）
2. ✅ GIoU：即使不重疊也有梯度（可以引導框移動）

#### b) **寬高Loss（尺寸損失）**

**計算：**
```python
# 對寬高取log後計算MSE
target_wh = torch.log(target_boxes[..., 2:4] + 1e-7)
pred_wh = torch.log(pred_boxes[..., 2:4] + 1e-7)
wh_loss = F.mse_loss(pred_wh, target_wh)
```

**為什麼用log：**
- 小物體：誤差1px影響很大（如10px→11px = 10%變化）
- 大物體：誤差1px影響很小（如100px→101px = 1%變化）
- **log空間**：平衡大小物體的損失貢獻

**總Box Loss：**
```python
box_loss = giou_loss + wh_loss
```

**如何解讀：**
- Box Loss = 0.0：預測框完全準確
- Box Loss = 1.0：預測框位置/大小偏差較大
- Box Loss < 0.5：預測框基本準確

---

### 2. **Obj Loss（物體置信度損失）**

**作用：**
- 判斷該位置**是否有物體**（不管是什麼物體）

**計算方法：**
```python
# yolo_loss.py Line 194-205
# 正樣本：有物體的格子
obj_loss_pos = focal_loss(pred_obj[pos_mask], target_obj[pos_mask])

# 負樣本：沒有物體的格子
obj_loss_neg = focal_loss(pred_obj[neg_mask], target_obj[neg_mask])

# 總損失（負樣本權重較低）
obj_loss = obj_loss_pos + lambda_noobj * obj_loss_neg
```

**包含兩部分：**

#### a) **正樣本Loss（有物體）**

- 目標：預測 confidence = IoU（與真實框的重疊度）
- 期望：模型對有物體的格子輸出高置信度

#### b) **負樣本Loss（無物體）**

- 目標：預測 confidence = 0
- 期望：模型對空白區域輸出低置信度
- 權重 λ_noobj = 0.5（降低負樣本影響）

**為什麼需要負樣本：**
- 圖片中大部分區域是背景（無物體）
- 模型需要學會區分「有物體」vs「無物體」
- 防止模型到處都預測有物體（False Positive）

**使用 Focal Loss：**
```python
# Focal Loss公式
FL(p) = -α(1-p)^γ log(p)
```

**優勢：**
- 簡單樣本（易分類）：權重降低（(1-p)^γ很小）
- 困難樣本（難分類）：權重提高（(1-p)^γ較大）
- **專注於難以區分的樣本**

**如何解讀：**
- Obj Loss = 0.0：完美區分有/無物體
- Obj Loss < 1.0：物體檢測基本準確
- Obj Loss > 3.0：模型混淆有無物體

---

### 3. **Class Loss（分類損失）**

**作用：**
- 預測物體的**具體類別**（20類：人、車、貓...）

**計算方法：**
```python
# yolo_loss.py Line 207-214
# 只對有物體的格子計算分類損失
cls_loss = F.binary_cross_entropy_with_logits(
    pred_cls[pos_mask],  # 預測的20個類別概率
    target_cls[pos_mask],  # 真實類別（one-hot編碼）
    reduction='mean'
)
```

**使用 Binary Cross Entropy（BCE）：**

**公式：**
```
BCE = -[y·log(p) + (1-y)·log(1-p)]
```

**其中：**
- y：真實標籤（0或1）
- p：預測概率（0~1）

**為什麼用BCE而非多類別CE：**
- YOLO採用**多標籤分類**（一個物體可能屬於多個類別）
- 例如："person" 也可能是 "athlete"
- BCE允許多個類別同時為正

**如何解讀：**
- Class Loss = 0.0：分類完全正確
- Class Loss < 0.5：分類基本準確
- Class Loss > 1.0：分類錯誤較多

---

### 4. **Total Loss（總損失）**

**計算公式：**
```python
# yolo_loss.py Line 221-225
total_loss = (
    lambda_coord * box_loss +      # 邊界框損失
    lambda_obj * obj_loss +        # 物體置信度損失
    lambda_class * cls_loss        # 分類損失
)
```

**權重設定：**
```python
lambda_coord = 5.0   # 邊界框權重（最高）
lambda_obj = 1.0     # 物體置信度權重
lambda_class = 1.0   # 分類權重
lambda_noobj = 0.5   # 負樣本權重（最低）
```

**為什麼設置不同權重：**
1. **Box Loss × 5.0**：位置準確性最重要
   - 框不準會導致完全錯誤的預測
   
2. **Obj Loss × 1.0**：檢測物體存在
   - 需要平衡正負樣本
   
3. **Class Loss × 1.0**：分類準確性
   - 在框準確的前提下再分類
   
4. **Noobj Loss × 0.5**：避免背景主導
   - 圖片中90%是背景，降低權重防止不平衡

**如何解讀：**
- Total Loss = 0.0：完美預測（不可能達到）
- Total Loss < 5.0：模型性能良好
- Total Loss < 10.0：模型基本可用
- Total Loss > 20.0：模型仍在學習初期

---

## 📈 評估指標解釋

### mAP (mean Average Precision)

**定義：**
- 所有類別的AP（Average Precision）的平均值

**AP計算流程：**

1. **收集所有預測框**
   - 對每個類別，收集所有預測框及其confidence

2. **按confidence排序**
   - 從高到低排序（最自信的預測在前）

3. **計算Precision-Recall曲線**
   ```
   Precision = TP / (TP + FP)  # 預測為正的樣本中，真正為正的比例
   Recall = TP / (TP + FN)     # 實際為正的樣本中，被預測為正的比例
   ```

4. **計算曲線下面積（AP）**
   ```python
   AP = ∫[0→1] Precision(Recall) dRecall
   ```

5. **對所有類別求平均（mAP）**
   ```python
   mAP = (1/20) Σ AP_i
   ```

**IoU閾值：**
- 通常使用 IoU = 0.5（PASCAL VOC標準）
- 預測框與真實框 IoU > 0.5 → 視為正確預測（TP）
- 預測框與真實框 IoU ≤ 0.5 → 視為錯誤預測（FP）

**mAP 分數含義：**
- mAP = 1.0：完美檢測（不可能）
- mAP = 0.75：非常優秀（競賽水平）
- mAP = 0.60：良好（及格水平）
- mAP = 0.45：基本可用（訓練初期）
- mAP = 0.20：很差（模型未學到東西）

---

### AP@0.5 vs AP@[0.5:0.95]

**AP@0.5（PASCAL VOC）：**
- IoU閾值 = 0.5
- 較寬鬆的標準
- 只要框覆蓋50%就算對

**AP@[0.5:0.95]（MS COCO）：**
- IoU閾值從0.5到0.95，間隔0.05
- 嚴格的標準
- 要求框非常準確

**本Lab使用AP@0.5（PASCAL VOC標準）**

---

## 🎯 如何判斷模型好壞

### 1. **觀察Loss曲線**

#### 理想情況：
```
Epoch 1:  Total Loss = 25.0
Epoch 5:  Total Loss = 15.0  ↓
Epoch 10: Total Loss = 10.0  ↓
Epoch 20: Total Loss = 7.5   ↓
Epoch 30: Total Loss = 6.0   ↓ (逐漸減小)
Epoch 40: Total Loss = 5.8   ↓
Epoch 50: Total Loss = 5.7   → (收斂)
```

#### 異常情況：

**A. Loss不下降：**
```
Epoch 1:  Total Loss = 25.0
Epoch 5:  Total Loss = 24.8
Epoch 10: Total Loss = 25.2
Epoch 20: Total Loss = 24.5  ← 沒有明顯下降
```
**可能原因：**
- 學習率太小（lr太低）
- 模型架構問題
- 數據問題（標註錯誤）

**B. Loss震盪：**
```
Epoch 1:  Total Loss = 25.0
Epoch 5:  Total Loss = 15.0  ↓
Epoch 10: Total Loss = 30.0  ↑↑↑
Epoch 15: Total Loss = 10.0  ↓↓↓
Epoch 20: Total Loss = 35.0  ↑↑↑  ← 劇烈震盪
```
**可能原因：**
- 學習率太大（lr太高）
- Batch size太小
- 梯度爆炸

**C. Loss突然變成NaN/Inf：**
```
Epoch 1:  Total Loss = 25.0
Epoch 5:  Total Loss = 15.0
Epoch 10: Total Loss = nan    ← 數值爆炸
```
**可能原因：**
- 梯度爆炸（需要梯度裁剪）
- 學習率過大
- 數值不穩定（log(0)等）

---

### 2. **對比訓練/驗證Loss**

#### 情況A：泛化良好
```
Epoch 10:
  Training Loss:   8.2  ←
  Validation Loss: 8.5  ← 差距小（0.3）
```
**解釋：** 模型學到通用特徵，泛化能力強

#### 情況B：輕微過擬合
```
Epoch 10:
  Training Loss:   7.0  ←
  Validation Loss: 9.0  ← 差距中等（2.0）
```
**解釋：** 模型開始記憶訓練集，但尚可接受

#### 情況C：嚴重過擬合
```
Epoch 10:
  Training Loss:   3.0  ←←←
  Validation Loss: 12.0  ← 差距大（9.0）
```
**解釋：** 模型完全記憶訓練集，泛化失敗

**解決方法：**
- 增加數據增強（Data Augmentation）
- 使用Dropout（目前未實現）
- 減少訓練epoch
- 增大訓練集

---

### 3. **觀察mAP趨勢**

#### 理想情況：
```
Epoch 5:  mAP = 0.35
Epoch 10: mAP = 0.45  ↑
Epoch 15: mAP = 0.52  ↑
Epoch 20: mAP = 0.58  ↑ (穩定提升)
Epoch 25: mAP = 0.61  ↑
Epoch 30: mAP = 0.63  ↑
```

#### 異常情況：

**A. mAP不提升：**
```
Epoch 5:  mAP = 0.35
Epoch 10: mAP = 0.36
Epoch 15: mAP = 0.35
Epoch 20: mAP = 0.37  ← 沒有進步
```
**可能原因：**
- 模型未學到有效特徵
- Loss雖然下降但預測不準確
- 需要調整NMS閾值

**B. mAP先升後降：**
```
Epoch 5:  mAP = 0.35
Epoch 10: mAP = 0.48  ↑
Epoch 15: mAP = 0.55  ↑ (最佳)
Epoch 20: mAP = 0.52  ↓
Epoch 25: mAP = 0.47  ↓ ← 下降
```
**可能原因：**
- **過擬合**：模型記憶訓練集
- 應該選擇Epoch 15的模型

---

### 4. **查看各類別AP**

**evaluate函數返回每個類別的AP：**

```python
val_aps = evaluate(model, eval_loader)
# val_aps = [0.65, 0.72, 0.45, ..., 0.58]  # 20個類別的AP

for i, ap in enumerate(val_aps):
    print(f"{CLASSES[i]}: {ap:.4f}")
```

**輸出示例：**
```
aeroplane: 0.7234
bicycle:   0.6521
bird:      0.5432
boat:      0.4123
...
```

**分析：**
- 某些類別AP特別低 → 該類別樣本少或難檢測
- 某些類別AP特別高 → 該類別特徵明顯

---

## 🔧 實戰建議

### 訓練初期（Epoch 1-10）

**期望：**
- Total Loss: 30 → 10
- mAP: 0.0 → 0.35

**觀察重點：**
- Loss是否下降
- 是否出現NaN

### 訓練中期（Epoch 10-30）

**期望：**
- Total Loss: 10 → 6
- mAP: 0.35 → 0.55

**觀察重點：**
- 訓練/驗證Loss差距
- mAP是否穩定提升

### 訓練後期（Epoch 30-50）

**期望：**
- Total Loss: 6 → 5.5（收斂）
- mAP: 0.55 → 0.62

**觀察重點：**
- 是否過擬合
- 是否需要提前停止

---

## 📝 評估檢查清單

訓練完成後，檢查以下項目：

- [ ] **Loss下降**：總損失從初始值下降60%以上
- [ ] **Loss收斂**：最後5個epoch損失變化<5%
- [ ] **無過擬合**：驗證Loss ≤ 訓練Loss × 1.3
- [ ] **mAP達標**：mAP > 0.50
- [ ] **無異常**：無NaN、無劇烈震盪
- [ ] **各分項Loss合理**：
  - Box Loss < 2.0
  - Obj Loss < 3.0
  - Class Loss < 1.0

---

## 🎓 總結

### 如何知道預測效果：

1. ✅ **訓練時mAP**：每5個epoch自動評估（已實現）
2. ✅ **驗證集Loss**：每個epoch顯示（已實現）
3. ✅ **測試集預測**：用predict_test.py生成result.csv
4. ✅ **視覺化**：繪製預測框（需自行實現）

### Loss意義總結：

| Loss類型 | 作用 | 理想值 | 可接受值 |
|---------|-----|--------|----------|
| **Box Loss** | 邊界框位置/大小 | <0.5 | <2.0 |
| **Obj Loss** | 物體置信度 | <0.5 | <3.0 |
| **Class Loss** | 類別分類 | <0.2 | <1.0 |
| **Total Loss** | 總損失 | <3.0 | <10.0 |

### 判斷模型好壞：

1. ✅ Loss持續下降 + 收斂
2. ✅ 訓練/驗證Loss接近
3. ✅ mAP穩定提升
4. ✅ mAP > 0.50

開始訓練後，按照本指南監控各項指標即可評估模型性能！🚀
