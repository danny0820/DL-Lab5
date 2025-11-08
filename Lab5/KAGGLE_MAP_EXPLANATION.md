# Kaggle mAP 評估腳本詳細分析

## 📋 文件概述

**文件路徑：** `Lab5/kaggle_map.py`

**用途：** Kaggle競賽的物體檢測評估指標計算腳本，實現 **VOC-style mAP @ IoU 0.5**

**主要功能：**
1. 解析CSV格式的預測結果和真實標籤
2. 計算每個類別的Average Precision (AP)
3. 返回所有類別的平均值 (mean AP)

---

## 🎯 核心概念

### mAP (mean Average Precision)

```
mAP = (1/20) × Σ AP_i
```

其中：
- 20個類別：PASCAL VOC數據集的20個物體類別
- AP_i：第i個類別的Average Precision

### IoU閾值

```python
IOU_THRESHOLD: float = 0.5
```

- **IoU ≥ 0.5**：預測框視為正確 (True Positive)
- **IoU < 0.5**：預測框視為錯誤 (False Positive)

---

## 📊 數據格式

### CSV文件結構

**Solution (真實標籤)：**
```csv
id,prediction_list
2007_000027,"[['person', 1.0, 174, 101, 349, 351], ['chair', 1.0, 6, 112, 362, 450]]"
```

**Submission (預測結果)：**
```csv
id,prediction_list
2007_000027,"[['person', 0.95, 170, 98, 352, 348], ['chair', 0.87, 5, 110, 365, 455]]"
```

### 預測列表格式

每個檢測框編碼為：
```python
['class_name', score, xmin, ymin, xmax, ymax]
```

**欄位說明：**
- `class_name` (str)：物體類別名稱（如 "person"）
- `score` (float)：置信度分數（0.0 ~ 1.0）
- `xmin, ymin` (float)：左上角坐標
- `xmax, ymax` (float)：右下角坐標

**示例：**
```python
['person', 0.95, 174.5, 101.3, 349.2, 351.8]
# 含義：檢測到一個"person"，置信度0.95，位於(174.5, 101.3)到(349.2, 351.8)
```

---

## 🔧 主要函數分析

### 1. `score()` - 主評估函數

**函數簽名：**
```python
def score(solution: pd.DataFrame, 
         submission: pd.DataFrame, 
         row_id_column_name: str) -> float
```

**功能：** 計算submission相對於solution的mAP分數

**處理流程：**

#### Step 1: 驗證數據完整性
```python
_validate_columns(solution, submission, row_id_column_name)
```
- 檢查必要欄位是否存在
- 檢查是否有重複的image_id
- 檢查submission是否包含所有必需的image_id

#### Step 2: 解析真實標籤
```python
gt_boxes, gt_class_counts = _parse_ground_truth(sol["prediction_list"])
```
- 解析每張圖片的真實框
- 統計每個類別的真實框數量

#### Step 3: 解析預測結果
```python
pred_by_class = _parse_predictions(sub["prediction_list"])
```
- 將預測結果按類別分組
- 保留image_id和confidence信息

#### Step 4: 計算每個類別的AP
```python
for class_idx in range(len(CLASSES)):
    ap = _average_precision_for_class(
        class_idx,
        gt_boxes,
        gt_class_counts[class_idx],
        pred_by_class[class_idx],
    )
    if ap is not None:
        aps.append(ap)
```

#### Step 5: 返回mAP
```python
result = float(np.mean(aps))
```

**返回值：**
- `float`：mAP分數（0.0 ~ 1.0）
- 如果結果非有限值（NaN/Inf），返回0.0

---

### 2. `_parse_ground_truth()` - 解析真實標籤

**函數簽名：**
```python
def _parse_ground_truth(series: pd.Series) -> Tuple[Dict, np.ndarray]
```

**功能：** 將CSV中的字符串轉換為可查詢的數據結構

**返回值：**

#### a) `gt_boxes: Dict[Tuple[str, int], List[np.ndarray]]`

**結構：**
```python
{
    ('2007_000027', 14): [array([174., 101., 349., 351.])],  # person
    ('2007_000027', 8):  [array([6., 112., 362., 450.])]      # chair
}
```

**鍵 (key)：** `(image_id, class_idx)`
**值 (value)：** 該圖片中該類別的所有真實框列表

#### b) `class_counts: np.ndarray`

**結構：**
```python
array([120, 85, 95, ...])  # 20個元素，每個是該類別的總框數
```

**用途：** 計算Recall時需要知道總共有多少個真實框

---

### 3. `_parse_predictions()` - 解析預測結果

**函數簽名：**
```python
def _parse_predictions(series: pd.Series) -> Dict[int, List[Tuple]]
```

**功能：** 將預測結果按類別分組

**返回值：**

```python
{
    14: [  # class_idx=14 (person)
        ('2007_000027', 0.95, array([170., 98., 352., 348.])),
        ('2007_000032', 0.87, array([...]))
    ],
    8: [   # class_idx=8 (chair)
        ('2007_000027', 0.87, array([5., 110., 365., 455.])),
        ...
    ]
}
```

**格式：** `{class_idx: [(image_id, score, box), ...]}`

**特點：**
- 所有預測都按類別分組
- 保留原始順序（稍後會按score排序）
- 驗證confidence在[0, 1]範圍內

---

### 4. `_decode_prediction_list()` - 解碼單個預測列表

**函數簽名：**
```python
def _decode_prediction_list(value: object, 
                           *, 
                           context: str) -> List[Tuple]
```

**功能：** 將字符串轉換為Python對象

**處理流程：**

#### Step 1: 處理空值
```python
if value is None or (isinstance(value, float) and np.isnan(value)):
    return []
```

#### Step 2: 解析字符串
```python
data = ast.literal_eval(stripped)
```
- 使用`ast.literal_eval()`安全地解析Python字面量
- 比`eval()`安全（不執行任意代碼）

#### Step 3: 驗證格式
```python
if len(det) != 6:
    raise ParticipantVisibleError(...)
```
- 每個檢測必須有6個元素
- `[class_name, score, xmin, ymin, xmax, ymax]`

#### Step 4: 驗證類別名稱
```python
if class_name not in CLASS_TO_INDEX:
    raise ParticipantVisibleError(f"Unknown class '{class_name}'.")
```

#### Step 5: 驗證邊界框
```python
if xmax_f < xmin_f or ymax_f < ymin_f:
    raise ParticipantVisibleError("Bounding box has negative area.")
```

**返回值：**
```python
[(class_idx, score, box), ...]
```

---

### 5. `_average_precision_for_class()` - 計算單個類別的AP

**函數簽名：**
```python
def _average_precision_for_class(
    class_idx: int,
    gt_boxes: Dict,
    num_gt: int,
    predictions: Sequence[Tuple]
) -> Optional[float]
```

**功能：** 實現VOC2010 AP計算協議

**處理流程：**

#### Step 1: 處理邊界情況
```python
if num_gt == 0:
    return None  # 該類別無真實框，不計入mAP
if not predictions:
    return 0.0   # 無預測，AP=0
```

#### Step 2: 按confidence排序預測
```python
sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
```
- 從高到低排序
- 優先處理置信度高的預測

#### Step 3: 初始化TP/FP數組
```python
tp = np.zeros(len(sorted_preds), dtype=np.float32)
fp = np.zeros(len(sorted_preds), dtype=np.float32)
```

#### Step 4: 標記已匹配的真實框
```python
gt_used: Dict[Tuple[str, int], np.ndarray] = {
    key: np.zeros(len(boxes), dtype=bool) 
    for key, boxes in gt_boxes.items() if key[1] == class_idx
}
```
- 每個真實框只能匹配一次（greedy matching）

#### Step 5: 匹配預測與真實框
```python
for i, (image_id, score, box) in enumerate(sorted_preds):
    key = (image_id, class_idx)
    gts = gt_boxes.get(key, [])
    if gts:
        # 計算與所有真實框的IoU
        overlaps = np.array([_bbox_iou(box, gt_box) for gt_box in gts])
        best = overlaps.argmax()
        best_iou = overlaps[best]
        
        # 判斷是否匹配成功
        if best_iou >= IOU_THRESHOLD and not gt_used[key][best]:
            tp[i] = 1.0  # True Positive
            gt_used[key][best] = True
        else:
            fp[i] = 1.0  # False Positive
    else:
        fp[i] = 1.0  # 該圖片無該類別真實框
```

**匹配規則：**
1. 找到IoU最大的真實框
2. IoU ≥ 0.5 且該真實框未被匹配 → TP
3. 否則 → FP

#### Step 6: 計算累積TP/FP
```python
tp = np.cumsum(tp)
fp = np.cumsum(fp)
```

**示例：**
```python
# 原始
tp = [1, 0, 1, 0, 1]
fp = [0, 1, 0, 1, 0]

# 累積
tp = [1, 1, 2, 2, 3]  # 到目前為止有多少TP
fp = [0, 1, 1, 2, 2]  # 到目前為止有多少FP
```

#### Step 7: 計算Precision和Recall
```python
recall = tp / num_gt
precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
```

**公式：**
```
Recall = TP / (TP + FN) = TP / num_gt
Precision = TP / (TP + FP)
```

**示例：**
```python
num_gt = 5  # 該類別共5個真實框
tp = [1, 1, 2, 2, 3]
fp = [0, 1, 1, 2, 2]

recall    = [0.2, 0.2, 0.4, 0.4, 0.6]
precision = [1.0, 0.5, 0.67, 0.5, 0.6]
```

#### Step 8: 計算AP
```python
return _voc_ap(recall, precision)
```

---

### 6. `_bbox_iou()` - 計算IoU

**函數簽名：**
```python
def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float
```

**功能：** 計算兩個邊界框的交並比 (Intersection over Union)

**計算步驟：**

#### Step 1: 計算交集區域
```python
ixmin = max(box_a[0], box_b[0])  # 交集左邊界
iymin = max(box_a[1], box_b[1])  # 交集上邊界
ixmax = min(box_a[2], box_b[2])  # 交集右邊界
iymax = min(box_a[3], box_b[3])  # 交集下邊界

iw = max(ixmax - ixmin + 1.0, 0.0)  # 交集寬度
ih = max(iymax - iymin + 1.0, 0.0)  # 交集高度
inter = iw * ih  # 交集面積
```

**為什麼+1.0？**
- 像素坐標是離散的
- 如果xmin=10, xmax=20，實際有11個像素（10, 11, ..., 20）
- 寬度 = xmax - xmin + 1 = 11

#### Step 2: 計算各框面積
```python
area_a = (box_a[2] - box_a[0] + 1.0) * (box_a[3] - box_a[1] + 1.0)
area_b = (box_b[2] - box_b[0] + 1.0) * (box_b[3] - box_b[1] + 1.0)
```

#### Step 3: 計算聯集和IoU
```python
union = area_a + area_b - inter
if union <= 0.0:
    return 0.0
return float(inter / union)
```

**公式：**
```
IoU = 交集面積 / 聯集面積
    = inter / (area_a + area_b - inter)
```

**圖示：**
```
Box A: [10, 10, 50, 50]  (面積 = 41×41 = 1681)
Box B: [30, 30, 70, 70]  (面積 = 41×41 = 1681)

交集: [30, 30, 50, 50]  (面積 = 21×21 = 441)
聯集: 1681 + 1681 - 441 = 2921

IoU = 441 / 2921 ≈ 0.15
```

---

### 7. `_voc_ap()` - VOC AP計算

**函數簽名：**
```python
def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float
```

**功能：** 使用VOC2010協議計算AP（11點插值法的改進版）

**處理步驟：**

#### Step 1: 添加邊界點
```python
mrec = np.concatenate(([0.0], recall, [1.0]))
mpre = np.concatenate(([0.0], precision, [0.0]))
```

**目的：** 確保曲線從(0,0)開始，到(1,0)結束

#### Step 2: 單調化Precision
```python
for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
```

**作用：** 將Precision-Recall曲線變為單調遞減

**示例：**
```python
# 原始
precision = [1.0, 0.5, 0.67, 0.5, 0.6]

# 單調化（從右往左取最大值）
precision = [1.0, 0.67, 0.67, 0.6, 0.6]
```

**為什麼這樣做？**
- 消除鋸齒狀波動
- 使用"右側最大值"作為插值
- VOC2010協議規定的標準做法

#### Step 3: 計算曲線下面積
```python
idx = np.where(mrec[1:] != mrec[:-1])[0]  # 找到Recall變化的位置
ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
```

**公式：**
```
AP = Σ (Recall_{i+1} - Recall_i) × Precision_{i+1}
```

**幾何意義：**
- 計算Precision-Recall曲線下的面積
- 使用矩形近似（數值積分）

**圖示：**
```
Precision
    ^
1.0 |█
    |█
0.8 |█   █
    |█   █
0.6 |█   █   █
    |█   █   █
    +-----------> Recall
    0  0.2 0.4 0.6
    
AP = 1.0×0.2 + 0.8×0.2 + 0.6×0.2 = 0.48
```

---

## 🎯 完整評估流程示例

### 輸入數據

**Solution (真實標籤)：**
```csv
id,prediction_list
img1,"[['person', 1.0, 10, 10, 50, 50], ['car', 1.0, 60, 60, 100, 100]]"
img2,"[['person', 1.0, 20, 20, 60, 60]]"
```

**Submission (預測結果)：**
```csv
id,prediction_list
img1,"[['person', 0.9, 12, 12, 52, 52], ['car', 0.8, 62, 62, 102, 102], ['person', 0.7, 150, 150, 200, 200]]"
img2,"[['person', 0.95, 22, 22, 62, 62]]"
```

### 處理流程

#### 1. 解析真實標籤
```python
gt_boxes = {
    ('img1', 14): [array([10, 10, 50, 50])],  # person
    ('img1', 6):  [array([60, 60, 100, 100])],  # car
    ('img2', 14): [array([20, 20, 60, 60])]   # person
}

class_counts = array([0, ..., 2, ..., 1, ...])  # person有2個, car有1個
```

#### 2. 解析預測結果
```python
pred_by_class = {
    14: [  # person
        ('img1', 0.9, array([12, 12, 52, 52])),
        ('img2', 0.95, array([22, 22, 62, 62])),
        ('img1', 0.7, array([150, 150, 200, 200]))
    ],
    6: [  # car
        ('img1', 0.8, array([62, 62, 102, 102]))
    ]
}
```

#### 3. 計算person類別的AP

**Step 3.1: 按confidence排序**
```python
sorted_preds = [
    ('img2', 0.95, array([22, 22, 62, 62])),  # 最高confidence
    ('img1', 0.9, array([12, 12, 52, 52])),
    ('img1', 0.7, array([150, 150, 200, 200]))
]
```

**Step 3.2: 匹配預測與真實框**

| 預測 | 真實框 | IoU | 匹配 | 結果 |
|-----|--------|-----|------|-----|
| img2, conf=0.95, [22,22,62,62] | img2, [20,20,60,60] | 0.82 | ✅ | TP |
| img1, conf=0.9, [12,12,52,52] | img1, [10,10,50,50] | 0.84 | ✅ | TP |
| img1, conf=0.7, [150,150,200,200] | img1, [10,10,50,50] | 0.0 | ❌ | FP |

**Step 3.3: 計算TP/FP**
```python
tp = [1, 1, 0] → cumsum → [1, 2, 2]
fp = [0, 0, 1] → cumsum → [0, 0, 1]

num_gt = 2

recall    = [1/2, 2/2, 2/2] = [0.5, 1.0, 1.0]
precision = [1/1, 2/2, 2/3] = [1.0, 1.0, 0.67]
```

**Step 3.4: 計算AP**
```python
# 添加邊界
mrec = [0.0, 0.5, 1.0, 1.0, 1.0]
mpre = [0.0, 1.0, 1.0, 0.67, 0.0]

# 單調化
mpre = [0.0, 1.0, 1.0, 1.0, 0.0]

# 計算面積
AP_person = (0.5-0.0)×1.0 + (1.0-0.5)×1.0 + (1.0-1.0)×1.0
         = 0.5 + 0.5 + 0.0
         = 1.0
```

#### 4. 計算car類別的AP

**類似流程：**
```python
# 只有1個預測，1個真實框
# IoU計算後匹配成功
AP_car = 1.0
```

#### 5. 計算mAP
```python
mAP = (AP_person + AP_car) / 2
    = (1.0 + 1.0) / 2
    = 1.0
```

---

## ⚠️ 異常處理

### ParticipantVisibleError

**定義：**
```python
class ParticipantVisibleError(Exception):
    """Raised for submission issues that the competitor can fix."""
```

**用途：** 向參賽者顯示可以修復的錯誤

### 常見錯誤

#### 1. 缺少必要欄位
```python
ParticipantVisibleError("Submission file missing columns: ['prediction_list']")
```

#### 2. 重複的image_id
```python
ParticipantVisibleError("Submission contains duplicated image ids.")
```

#### 3. 缺少預測
```python
ParticipantVisibleError("Submission is missing predictions for ids: ['img1', 'img2']")
```

#### 4. 未知類別
```python
ParticipantVisibleError("Unknown class 'dog'.")
```

#### 5. 非法confidence
```python
ParticipantVisibleError("Invalid confidence score -0.5 for image img1.")
```

#### 6. 負面積邊界框
```python
ParticipantVisibleError("Bounding box has negative area.")
```

---

## 📝 與 predict_test.py 的關聯

### predict_test.py 生成的格式

```python
# predict_test.py 輸出
result = {
    'image_id': ['2007_000027', '2007_000032', ...],
    'label_list': [
        '14 0.315 0.229 0.445 0.478 0.972',  # class_id x1 y1 x2 y2 conf
        '11 0.156 0.129 0.712 0.893 0.965',
        ...
    ]
}
```

### 需要轉換為 kaggle_map.py 格式

**轉換腳本示例：**
```python
import pandas as pd

# 讀取 predict_test.py 的輸出
df = pd.read_csv('result.csv')

# 轉換格式
def convert_to_kaggle_format(row):
    detections = []
    for det in row['label_list'].split(';'):
        if det:
            class_id, x1, y1, x2, y2, conf = det.split()
            class_name = CLASSES[int(class_id)]
            detections.append([
                class_name, 
                float(conf), 
                float(x1), 
                float(y1), 
                float(x2), 
                float(y2)
            ])
    return str(detections)

df['prediction_list'] = df.apply(convert_to_kaggle_format, axis=1)
df = df[['image_id', 'prediction_list']]
df.to_csv('submission.csv', index=False)
```

---

## 🎓 關鍵技術點總結

### 1. 貪心匹配策略 (Greedy Matching)

- 按confidence從高到低處理預測
- 每個真實框只能匹配一次
- 高confidence預測優先搶佔真實框

**優點：**
- ✅ 簡單高效
- ✅ 符合VOC協議

**缺點：**
- ⚠️ 非最優匹配（不是全局最優）
- ⚠️ 後續低confidence預測可能找不到匹配

### 2. Precision-Recall 曲線

**特點：**
- Recall單調遞增（隨著預測增多）
- Precision通常震盪（TP/FP比例變化）

**單調化的意義：**
- 使用"右側最大值"平滑曲線
- 消除局部波動
- 標準化評估方法

### 3. VOC2010 vs VOC2007

**VOC2007 (11點插值)：**
```python
ap = 0
for t in [0, 0.1, 0.2, ..., 1.0]:
    ap += max(precision where recall >= t)
ap /= 11
```

**VOC2010 (本腳本使用)：**
```python
# 使用所有Recall變化點
ap = Σ (recall[i+1] - recall[i]) × precision[i+1]
```

**優勢：**
- ✅ 更精確（使用所有數據點）
- ✅ 不受固定插值點限制

### 4. IoU計算的+1.0

```python
area = (xmax - xmin + 1.0) * (ymax - ymin + 1.0)
```

**原因：**
- 像素坐標是離散的
- 邊界框包含端點
- 符合VOC標準

### 5. 數值穩定性

```python
precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
```

**目的：** 避免除以0

```python
if not np.isfinite(result):
    result = 0.0
```

**目的：** 處理NaN/Inf

---

## 🔍 調試技巧

### 1. 查看各類別AP

```python
for i, cls in enumerate(CLASSES):
    ap = aps[i] if i < len(aps) else None
    print(f"{cls}: {ap:.4f}" if ap is not None else f"{cls}: N/A")
```

### 2. 查看匹配結果

```python
for i, (image_id, score, box) in enumerate(sorted_preds):
    result = "TP" if tp[i] == 1.0 else "FP"
    print(f"Pred {i}: {result}, conf={score:.3f}, img={image_id}")
```

### 3. 驗證IoU計算

```python
box_a = np.array([10, 10, 50, 50])
box_b = np.array([30, 30, 70, 70])
iou = _bbox_iou(box_a, box_b)
print(f"IoU: {iou:.4f}")
```

---

## 📊 性能考量

### 時間複雜度

**總體：** O(N × M)
- N：預測框總數
- M：每張圖片的真實框數量（通常很小）

**瓶頸：** IoU計算（雙重循環）

### 空間複雜度

**O(N + K)**
- N：存儲所有預測
- K：存儲所有真實框

---

## 🎯 實際使用

### 本地評估

```python
import pandas as pd
from kaggle_map import score

# 載入數據
solution = pd.read_csv('solution.csv')
submission = pd.read_csv('submission.csv')

# 計算mAP
map_score = score(solution, submission, 'id')
print(f"mAP: {map_score:.4f}")
```

### Kaggle提交

1. 使用 `predict_test.py` 生成預測
2. 轉換為Kaggle格式
3. 上傳 `submission.csv` 到Kaggle
4. Kaggle使用 `kaggle_map.py` 自動評分

---

## 📚 參考資料

### VOC Challenge

- **論文：** "The PASCAL Visual Object Classes Challenge 2010 (VOC2010)"
- **網站：** http://host.robots.ox.ac.uk/pascal/VOC/
- **標準：** IoU閾值0.5，11點插值改進版

### 相關概念

- **IoU (Intersection over Union)**：交並比
- **AP (Average Precision)**：平均精度
- **mAP (mean Average Precision)**：平均AP
- **TP/FP/FN**：真陽/假陽/假陰
- **Precision/Recall**：精確率/召回率

---

## ✅ 總結

### kaggle_map.py 的作用

1. ✅ **標準化評估**：使用VOC2010協議
2. ✅ **自動化評分**：Kaggle競賽後台使用
3. ✅ **錯誤檢測**：驗證提交格式
4. ✅ **公平比較**：所有參賽者使用相同評估方法

### 關鍵特點

- **IoU閾值**：0.5
- **匹配策略**：貪心匹配
- **AP計算**：VOC2010插值法
- **mAP**：20個類別AP的平均

### 與訓練的關聯

- **訓練時**：使用 `src/eval_voc.py` 評估mAP
- **提交時**：Kaggle使用 `kaggle_map.py` 評估
- **兩者應該一致**：確保本地評估準確

理解這個腳本有助於：
1. ✅ 了解評估標準
2. ✅ 調試預測結果
3. ✅ 優化模型性能
4. ✅ 準備Kaggle提交

🚀 現在您可以準確理解模型是如何被評估的！
