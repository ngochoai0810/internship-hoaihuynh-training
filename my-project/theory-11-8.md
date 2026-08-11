# Buổi 2 — ML Techniques Overview + Regularization (Ridge & Lasso)

> Bản diễn giải dễ hiểu, dùng để tự học trước khi dựng notebook `02_regularized_regression.ipynb`.

---

## 1. Core ML Concepts

### 1.1 Supervised vs Unsupervised Learning

*Giải thích thêm: "labeled data" nghĩa là dữ liệu đã có sẵn đáp án đúng đi kèm — giống như bộ đề có đáp án ở cuối sách, mô hình học bằng cách so đáp án của mình với đáp án đúng rồi tự sửa. "Ground truth" chính là đáp án đúng đó.*

**Bảng 1 — Supervised vs Unsupervised Learning**

| Tiêu chí | Supervised | Unsupervised |
|---|---|---|
| Training data | Có nhãn (input + đúng output) | Không có nhãn |
| Goal | Dự đoán output cho dữ liệu mới | Tìm cấu trúc/nhóm ẩn |
| Evaluation | So khớp ground truth (RMSE, accuracy...) | Khó đánh giá khách quan (silhouette, inertia...) |
| Example | Dự đoán giá nhà, phân loại spam | Phân khúc khách hàng, phát hiện bất thường |
| Algorithms | Linear/Ridge/Lasso, Logistic Regression, Random Forest, SVM | K-means, Hierarchical Clustering, PCA, GMM |

### 1.2 Regression vs Classification

*Phân biệt đơn giản: nếu câu trả lời là "bao nhiêu" (một con số) → regression. Nếu câu trả lời là "loại nào" (chọn 1 trong các lựa chọn có sẵn) → classification.*

**Bảng 2 — Regression vs Classification**

| Tiêu chí | Regression | Classification |
|---|---|---|
| Output | Giá trị liên tục (số bất kỳ) | Nhãn rời rạc/danh mục cố định |
| Ví dụ | Giá nhà, doanh thu, nhiệt độ | Spam/không spam, chó/mèo/chim |
| Đánh giá | RMSE, MAE, R² | Accuracy, Precision, Recall, F1 |

### 1.3 Vì sao House Price Prediction là Supervised Regression

1. **Có ground truth rõ ràng**: cột `SalePrice` là giá bán thật → có đáp án để đối chiếu → supervised.
2. **`SalePrice` là số liên tục**: có thể là bất kỳ giá trị dương nào, không phải nhãn cố định → regression, không phải classification.
3. **Cách đánh giá khác nhau**: regression đo khoảng cách số học (RMSE/MAE/R²), classification đo tỷ lệ đoán đúng nhãn (accuracy/precision/recall/F1).

---

## 2. Regularization Concepts

### 2.1 Vấn đề: Overfitting

*Ví von dễ hiểu: overfitting giống như học vẹt để thi — học thuộc lòng từng câu trong đề cũ (train set) đến mức làm đúng 100%, nhưng gặp đề mới (test set) thì làm sai vì không thực sự hiểu bản chất, chỉ nhớ chi tiết vụn vặt/nhiễu.*

Khi có quá nhiều feature hoặc multicollinearity (feature tương quan mạnh với nhau, vd. `GrLivArea` và `TotalBsmtSF` cùng phản ánh diện tích), Linear Regression dễ:

- Gán hệ số (coefficient) quá lớn để khớp hoàn hảo với train.
- Học luôn cả noise (nhiễu ngẫu nhiên) thay vì xu hướng chung.
- Kết quả: RMSE train thấp nhưng RMSE test cao.

### 2.2 Ý tưởng Regularization

*Hiểu đơn giản: regularization giống như "phạt điểm" mô hình nếu nó cố dùng hệ số quá lớn để chạy theo từng chi tiết nhỏ trong dữ liệu train — buộc mô hình phải "học tổng quát" thay vì "học vẹt".*

```
Loss (có regularization) = Loss gốc (sai số dự đoán) + alpha × penalty(coefficients)
```

`alpha` là hyperparameter kiểm soát mức phạt — alpha càng lớn, mô hình càng bị ép giữ hệ số nhỏ.

### 2.3 Ridge Regression (L2)

- Penalty: `alpha × Σ(coef²)` — tổng bình phương các hệ số.
- Hiệu ứng: co tất cả hệ số về gần 0, **không đưa hẳn về 0** — mọi feature vẫn được giữ lại.
- Phù hợp khi có multicollinearity (Ridge chia đều "trọng số" giữa các feature tương quan).

Code signature:

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
```

### 2.4 Lasso Regression (L1)

- Penalty: `alpha × Σ|coef|` — tổng trị tuyệt đối các hệ số.
- Hiệu ứng: có thể đưa hệ số về **đúng bằng 0** → loại bỏ hẳn feature đó khỏi mô hình.
- Hoạt động như feature selection tự động (dễ loại các feature nhiễu như `MiscFeature`, `PoolQC`).

Code signature:

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
```

*Vì sao L1 tạo ra 0 còn L2 thì không — hiểu trực giác: vùng ràng buộc của L1 là hình thoi có góc nhọn nằm ngay trên trục tọa độ, còn L2 là hình tròn trơn không góc. Điểm tối ưu "chạm" vào góc nhọn (nơi hệ số = 0) dễ xảy ra hơn nhiều so với chạm vào một điểm bất kỳ trên đường tròn — không cần chứng minh toán, chỉ cần nhớ hình ảnh này.*

**Bảng 3 — Ridge vs Lasso**

| Tiêu chí | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty formula | Σ coef² | Σ \|coef\| |
| Coefficient = 0? | Không | Có (feature selection) |
| Phù hợp khi | Nhiều feature tương quan cao | Nhiều feature nhiễu, muốn model gọn |
| Diễn giải | Khó hơn (giữ mọi feature) | Dễ hơn (chỉ còn feature quan trọng) |
| sklearn class | `Ridge` | `Lasso` |

### 2.5 Alpha ảnh hưởng thế nào

**Bảng 4 — Ảnh hưởng của Alpha**

| Alpha | Hành vi | Rủi ro |
|---|---|---|
| Rất nhỏ (≈0) | Gần giống Linear Regression thường | Dễ overfit |
| Vừa phải | Cân bằng fit tốt và hệ số nhỏ | Vùng lý tưởng |
| Rất lớn | Hệ số ép về gần 0, model gần như chỉ dự đoán mean | Dễ underfit |

### 2.6 Lưu ý kỹ thuật quan trọng

Bắt buộc **chuẩn hóa (standardize)** feature trước khi train Ridge/Lasso — vì penalty tính trên độ lớn tuyệt đối của hệ số, feature nào scale lớn (vd. `GrLivArea` hàng nghìn sqft) sẽ bị "phạt" không công bằng so với feature scale nhỏ (vd. `OverallQual` 1–10).

Code signature:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 3. Implementation Structure — `02_regularized_regression.ipynb`

| Section | Mục đích | Code signature |
|---|---|---|
| 0. Setup | Load lại X_train/X_test/y_train/y_test từ Buổi 1 (tránh split lại gây data leakage) | `joblib.load("session1_split.pkl")` |
| 1. Scaling | Chuẩn hóa feature bắt buộc cho Ridge/Lasso | `StandardScaler().fit_transform(X_train)` |
| 2. Baseline train | Train Ridge & Lasso alpha=1.0 để kiểm tra pipeline chạy đúng | `Ridge(alpha=1.0).fit(X_train_scaled, y_train)` |
| 3. Alpha sweep + regularization path | Sweep alpha (0.01→100), vẽ 2 biểu đồ coefficient theo alpha để so sánh trực quan Ridge vs Lasso | `for a in alphas: Ridge(alpha=a).fit(...)` |
| 4. Đếm zero coefficients | Đếm & liệt kê feature bị Lasso loại (coef=0) theo từng alpha | `np.sum(coefs == 0)` |
| 5. So sánh 4 model | Bảng RMSE/MAE/R² của Dummy/Linear/Ridge/Lasso | `evaluate(model, X_test, y_test, name)` |