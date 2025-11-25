# Deep Learning Training Script

Hướng dẫn sử dụng script `deep_prediction_main.py` để huấn luyện mô hình Temporal Fusion Transformer (TFT).

## Tổng Quan

Script này triển khai Phase 5 của TFT roadmap:
- Parse CLI arguments: symbol filter, timeframe, epochs, batch size, GPU flag, phase selection
- Build dataset/datamodule, instantiate TFT, và chạy `pl.Trainer`
- Log metrics to TensorBoard; track validation loss, MAE/RMSE, class accuracy
- Save: best checkpoint, dataset metadata, scaler/config JSON

## Cài Đặt

Đảm bảo đã cài đặt các dependencies:

```bash
pip install torch torchvision torchaudio pytorch-lightning pytorch-forecasting tensorboard optuna
```

## Sử Dụng Cơ Bản

### Phase 1: Vanilla TFT (MVP)

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT ETH/USDT \
    --timeframe 1h \
    --limit 2000 \
    --phase 1 \
    --epochs 100 \
    --batch-size 64
```

### Phase 2: Optuna Optimization

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT \
    --timeframe 1h \
    --phase 2 \
    --optuna-trials 20 \
    --optuna-max-epochs 50 \
    --epochs 100
```

### Phase 3: Hybrid LSTM + TFT

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT ETH/USDT \
    --timeframe 4h \
    --phase 3 \
    --epochs 100 \
    --task-type regression
```

## Các Tham Số CLI

### Data Arguments

- `-s, --symbols`: Danh sách trading pairs (mặc định: BTC/USDT)
- `-q, --quote`: Quote currency (mặc định: USDT)
- `-t, --timeframe`: Timeframe cho OHLCV data (mặc định: 1h)
- `-l, --limit`: Số lượng candles để fetch (mặc định: 1500)
- `-e, --exchanges`: Danh sách exchanges, phân cách bởi dấu phẩy

### Model Arguments

- `--phase`: Phase của model (1, 2, hoặc 3, mặc định: 1)
  - Phase 1: Vanilla TFT
  - Phase 2: Optuna Optimization
  - Phase 3: Hybrid LSTM + TFT
- `--task-type`: Loại task (regression hoặc classification, mặc định: regression)

### Training Arguments

- `--epochs`: Số epochs tối đa (mặc định: 100)
- `--batch-size`: Batch size (mặc định: 64)
- `--gpu`: Sử dụng GPU nếu có
- `--no-gpu`: Force sử dụng CPU
- `--gpus`: Số lượng GPUs để sử dụng

### Phase 2 (Optuna) Arguments

- `--optuna-trials`: Số lượng Optuna trials (mặc định: 20)
- `--optuna-max-epochs`: Số epochs tối đa mỗi trial (mặc định: 50)

### Output Arguments

- `--output-dir`: Thư mục output cho checkpoints và logs (mặc định: artifacts/deep)
- `--experiment-name`: Tên experiment cho TensorBoard (mặc định: auto-generated)

### Feature Selection

- `--no-feature-selection`: Tắt feature selection

## Ví Dụ Sử Dụng

### 1. Training đơn giản với Phase 1

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT \
    --timeframe 1h \
    --limit 2000 \
    --phase 1 \
    --epochs 50
```

### 2. Training với GPU

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT ETH/USDT \
    --timeframe 4h \
    --phase 1 \
    --gpu \
    --epochs 100 \
    --batch-size 128
```

### 3. Optuna Optimization

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT \
    --timeframe 1h \
    --phase 2 \
    --optuna-trials 30 \
    --optuna-max-epochs 40 \
    --epochs 100
```

### 4. Classification Task

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT \
    --timeframe 1h \
    --phase 1 \
    --task-type classification \
    --epochs 100
```

### 5. Multi-symbol Training

```bash
python deep_prediction_main.py \
    --symbols BTC/USDT ETH/USDT BNB/USDT \
    --timeframe 1h \
    --limit 3000 \
    --phase 1 \
    --epochs 100
```

## Output Files

Sau khi training, các files sau sẽ được tạo trong `--output-dir`:

1. **Checkpoints**: `checkpoints/tft-epoch=XX-val_loss=X.XXXX.ckpt`
   - Best model checkpoints (top K theo validation loss)

2. **Dataset Metadata**: `dataset_metadata.pkl`
   - Metadata của TimeSeriesDataSet để sử dụng cho inference

3. **Model Config**: `model_config.json`
   - Cấu hình của model (hyperparameters)

4. **Training Config**: `training_config.json`
   - Cấu hình training (phase, task type, symbols, etc.)

5. **TensorBoard Logs**: `logs/{experiment_name}/`
   - Logs cho TensorBoard visualization

## Xem TensorBoard Logs

```bash
tensorboard --logdir artifacts/deep/logs
```

Sau đó mở browser tại `http://localhost:6006`

## Metrics Tracked

Script tự động track các metrics sau trong TensorBoard:

- **Validation Loss**: `val_loss`
- **Training Loss**: `train_loss`
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error (nếu có)
- **Classification Accuracy**: `val_class_acc` (nếu task_type=classification)
- **Learning Rate**: `learning_rate`

## Lưu Ý

1. **GPU**: Script tự động detect GPU. Sử dụng `--gpu` để force GPU hoặc `--no-gpu` để force CPU.

2. **Memory**: Training với nhiều symbols hoặc batch size lớn có thể cần nhiều RAM/VRAM.

3. **Time**: Phase 2 (Optuna) có thể mất nhiều thời gian vì chạy nhiều trials.

4. **Feature Selection**: Mặc định bật. Tắt bằng `--no-feature-selection` nếu muốn sử dụng tất cả features.

5. **Multi-symbol Training**: TFT hỗ trợ training trên nhiều symbols cùng lúc. Đảm bảo có đủ data cho mỗi symbol.

## Troubleshooting

### Lỗi: "Training dataset is empty"
- Tăng `--limit` để fetch nhiều data hơn
- Kiểm tra xem symbols có hợp lệ không

### Lỗi: "CUDA out of memory"
- Giảm `--batch-size`
- Giảm số lượng symbols
- Sử dụng `--no-gpu` để train trên CPU

### Phase 2 chạy quá lâu
- Giảm `--optuna-trials`
- Giảm `--optuna-max-epochs`
- Sử dụng GPU để tăng tốc

## Next Steps

Sau khi training xong, bạn có thể:
1. Sử dụng checkpoints để inference (Phase 6: Inference Wrapper)
2. Evaluate model trên test set (Phase 7: Evaluation & Backtesting)
3. Deploy model cho production (Phase 8: Operationalization)

