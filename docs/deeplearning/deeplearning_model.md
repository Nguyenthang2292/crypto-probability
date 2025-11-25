# Cấu Hình Mô Hình Deep Learning

Tài liệu này mô tả việc triển khai 3 giai đoạn của mô hình Temporal Fusion Transformer (TFT) để dự đoán giá cryptocurrency.

## Tổng Quan

Cấu hình mô hình được triển khai trong `modules/deeplearning_model.py` với ba giai đoạn:

1. **Phase 1: Vanilla TFT (MVP)** - TFT chuẩn với QuantileLoss
2. **Phase 2: Optuna Optimization** - Tối ưu hyperparameters
3. **Phase 3: Hybrid LSTM + TFT** - Kiến trúc dual-branch nâng cao

## Phase 1: Vanilla TFT (MVP)

### Cách Sử Dụng

```python
from modules.deeplearning_model import create_vanilla_tft, create_training_callbacks
from modules.deeplearning_dataset import TFTDataModule
import pytorch_lightning as pl

# Tạo datamodule (giả sử train_df, val_df đã được chuẩn bị)
datamodule = TFTDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)

# Chuẩn bị dữ liệu và setup
datamodule.prepare_data()
datamodule.setup("fit")

# Tạo mô hình vanilla TFT
model = create_vanilla_tft(
    training_dataset=datamodule.training,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    learning_rate=0.03,
    quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
)

# Tạo callbacks
callbacks = create_training_callbacks(
    checkpoint_dir="artifacts/deep/checkpoints",
    monitor="val_loss",
    patience=10,
)

# Tạo trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    accelerator="auto",
    devices=1,
)

# Huấn luyện
trainer.fit(model, datamodule)
```

### Tính Năng Chính

- **QuantileLoss**: Tạo khoảng tin cậy (chỉ giao dịch khi CI hẹp)
- **Callbacks**: EarlyStopping, ModelCheckpoint, LearningRateMonitor
- **Hyperparameters**: Có thể cấu hình hidden_size, attention_head_size, dropout, learning_rate

### Cấu Hình

Các hyperparameters mặc định được định nghĩa trong `modules/config.py`:

```python
DEEP_MODEL_HIDDEN_SIZE = 16
DEEP_MODEL_ATTENTION_HEAD_SIZE = 4
DEEP_MODEL_DROPOUT = 0.1
DEEP_MODEL_LEARNING_RATE = 0.03
DEEP_MODEL_QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
```

## Phase 2: Tối Ưu Optuna

### Cách Sử Dụng

```python
from modules.deeplearning_model import optimize_tft_hyperparameters
from modules.deeplearning_dataset import TFTDataModule
import pytorch_lightning as pl

# Tạo datamodule
datamodule = TFTDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)
datamodule.prepare_data()
datamodule.setup("fit")

# Tối ưu hyperparameters
best_params, study = optimize_tft_hyperparameters(
    training_dataset=datamodule.training,
    val_dataset=datamodule.validation,
    datamodule=datamodule,
    n_trials=20,
    timeout=None,  # Không giới hạn thời gian
    n_jobs=1,
    study_name="tft_optimization",
    checkpoint_dir="artifacts/deep/optuna",
    max_epochs=50,
)

# Sử dụng tham số tốt nhất để tạo mô hình cuối cùng
from modules.deeplearning_model import create_vanilla_tft
final_model = create_vanilla_tft(
    training_dataset=datamodule.training,
    **best_params,
)

# Huấn luyện mô hình cuối cùng với hyperparameters tốt nhất
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=create_training_callbacks(),
)
trainer.fit(final_model, datamodule)
```

### Tính Năng Chính

- **TPE Sampler**: Tree-structured Parzen Estimator để tìm kiếm hiệu quả
- **Pruning**: Dừng sớm các thử nghiệm không hứa hẹn
- **Parallel Trials**: Hỗ trợ tối ưu song song (n_jobs > 1)

### Cấu Hình

```python
DEEP_OPTUNA_N_TRIALS = 20
DEEP_OPTUNA_TIMEOUT = None
DEEP_OPTUNA_N_JOBS = 1
DEEP_OPTUNA_MAX_EPOCHS = 50
```

### Không Gian Tìm Kiếm Hyperparameter

- `hidden_size`: 8-64 (bước 8)
- `attention_head_size`: 1-8
- `dropout`: 0.05-0.3 (bước 0.05)
- `learning_rate`: 1e-4 đến 0.1 (thang log)
- `reduce_on_plateau_patience`: 2-8

## Phase 3: Kiến Trúc Hybrid LSTM + TFT

### Kiến Trúc

```
Dữ Liệu Đầu Vào
    │
    ├─→ Nhánh LSTM (OHLCV thô) ──┐
    │                              │
    └─→ Nhánh TFT (Features phức tạp) ──┤
                                        │
                                    Gated Fusion (GLU)
                                        │
                                    Multi-task Head
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            Classification Head                    Regression Head
            (Hướng: UP/DOWN/NEUTRAL)        (Độ lớn: Quantiles)
```

### Cách Sử Dụng

```python
from modules.deeplearning_model import create_hybrid_lstm_tft
from modules.deeplearning_dataset import TFTDataModule
import pytorch_lightning as pl

# Tạo datamodule
datamodule = TFTDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)
datamodule.prepare_data()
datamodule.setup("fit")

# Tạo mô hình hybrid
model = create_hybrid_lstm_tft(
    training_dataset=datamodule.training,
    tft_hidden_size=16,
    tft_attention_head_size=4,
    tft_dropout=0.1,
    lstm_hidden_size=32,
    lstm_num_layers=2,
    fusion_size=64,
    num_classes=3,  # UP, NEUTRAL, DOWN
    quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    lambda_class=1.0,  # Trọng số cho classification loss
    lambda_reg=1.0,    # Trọng số cho regression loss
    learning_rate=0.001,
)

# Tạo callbacks
callbacks = create_training_callbacks(
    checkpoint_dir="artifacts/deep/checkpoints",
    monitor="val_loss",
)

# Huấn luyện
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
)
trainer.fit(model, datamodule)
```

### Tính Năng Chính

- **Dual Branch**:
  - **Nhánh LSTM**: Xử lý chuỗi giá/khối lượng thô (OHLCV)
  - **Nhánh TFT**: Xử lý features phức tạp (static + known future)
- **Gated Fusion (GLU)**: Kết hợp các latent vectors từ LSTM và TFT
- **Multi-task Head**:
  - **Task 1**: Hướng (Classification/Softmax) - UP/DOWN/NEUTRAL
  - **Task 2**: Độ lớn (Regression/QuantileLoss) - Khoảng tin cậy
- **Combined Loss**: λ_class * CrossEntropy + λ_reg * QuantileLoss

### Cấu Hình

```python
DEEP_HYBRID_LSTM_HIDDEN_SIZE = 32
DEEP_HYBRID_LSTM_NUM_LAYERS = 2
DEEP_HYBRID_FUSION_SIZE = 64
DEEP_HYBRID_NUM_CLASSES = 3
DEEP_HYBRID_LAMBDA_CLASS = 1.0
DEEP_HYBRID_LAMBDA_REG = 1.0
DEEP_HYBRID_LEARNING_RATE = 0.001
```

## Nạp và Lưu Mô Hình

### Lưu Cấu Hình Mô Hình

```python
from modules.deeplearning_model import save_model_config

save_model_config(
    model=model,
    config_path="artifacts/deep/model_config.json",
)
```

### Nạp Mô Hình Đã Huấn Luyện

```python
from modules.deeplearning_model import load_tft_model

# Nạp vanilla TFT
model = load_tft_model(
    checkpoint_path="artifacts/deep/checkpoints/tft-epoch=50-val_loss=0.0123.ckpt",
    training_dataset=datamodule.training,  # Bắt buộc cho TFT
)
```

## Cấu Hình Huấn Luyện

Các thiết lập huấn luyện chung trong `modules/config.py`:

```python
DEEP_MAX_EPOCHS = 100
DEEP_ACCELERATOR = "auto"  # 'auto', 'gpu', 'cpu'
DEEP_DEVICES = 1
DEEP_PRECISION = 32  # 16, 32, hoặc 'bf16'
DEEP_GRADIENT_CLIP_VAL = 0.5
DEEP_EARLY_STOPPING_PATIENCE = 10
DEEP_CHECKPOINT_SAVE_TOP_K = 3
```

## Chiến Lược Chọn Phase

1. **Bắt đầu với Phase 1 (Vanilla TFT)**: 
   - Nhanh chóng để triển khai và huấn luyện
   - Hiệu suất baseline tốt
   - Sử dụng QuantileLoss cho khoảng tin cậy

2. **Chuyển sang Phase 2 (Optuna) nếu Phase 1 đạt ngưỡng**:
   - Tự động tìm hyperparameters tối ưu
   - Có thể cải thiện hiệu suất đáng kể
   - Yêu cầu nhiều tài nguyên tính toán hơn

3. **Xem xét Phase 3 (Hybrid) chỉ khi Phase 2 đạt ngưỡng**:
   - Kiến trúc phức tạp nhất
   - Yêu cầu dữ liệu multi-task (classification + regression labels)
   - Tốt nhất cho các tình huống cần cả dự đoán hướng và độ lớn

## Lưu Ý

- **QuantileLoss**: Tạo khoảng tin cậy. Chỉ giao dịch khi CI hẹp (độ tin cậy cao).
- **Huấn Luyện Đa Tài Sản**: TFT hỗ trợ huấn luyện đa tài sản qua `group_ids=["symbol"]`.
- **Dữ Liệu Thiếu**: Dataset xử lý các nến bị thiếu qua resampling/interpolation (xem `deeplearning_dataset.py`).
- **Lựa Chọn Features**: Sử dụng feature selection trước khi huấn luyện (xem `feature_selection.py`).

## Dependencies

Các gói cần thiết:
- `torch`, `torchvision`, `torchaudio`
- `pytorch-lightning`
- `pytorch-forecasting`
- `tensorboard`
- `optuna` (tùy chọn, cho Phase 2)

Cài đặt bằng:
```bash
pip install torch torchvision torchaudio pytorch-lightning pytorch-forecasting tensorboard optuna
```
