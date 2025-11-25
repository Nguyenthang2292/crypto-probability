# Deep Learning Documentation

Tài liệu cho deep learning prediction component.

## Overview

Deep Learning component sử dụng Temporal Fusion Transformer (TFT) để dự đoán giá cryptocurrency với deep learning.

## Components

### Data Pipeline
**[deeplearning_data_pipeline.md](./deeplearning_data_pipeline.md)**

Pipeline chuẩn bị data cho TFT:
- Fractional differentiation
- Triple barrier method labeling
- Feature engineering
- Data scaling

### Dataset
- **Location:** `modules/deeplearning/dataset.py`
- Tạo TimeSeriesDataSet và DataLoaders:
  - TimeSeriesDataSet creation
  - DataModule cho PyTorch Lightning
  - Train/Val/Test splits

### Model
**[deeplearning_model.md](./deeplearning_model.md)**

TFT model architecture:
- Temporal Fusion Transformer implementation
- Attention mechanisms
- Quantile regression

### Training
**[deep_prediction_training.md](./deep_prediction_training.md)**

Hướng dẫn training:
- Training workflow
- Hyperparameter tuning
- Model checkpointing

### Feature Selection
**[feature_selection.md](./feature_selection.md)**

Chọn lọc và kỹ thuật hóa features:
- Mutual Information
- Boruta-like method
- F-test (ANOVA)
- Combined methods

## Usage

```bash
python deeplearning_prediction_main.py
```

## Configuration

Tất cả config được định nghĩa trong `modules/config.py` section **Deep Learning Configuration**:
- `DEEP_MAX_ENCODER_LENGTH` - Lookback window
- `DEEP_BATCH_SIZE` - Batch size
- `DEEP_MODEL_HIDDEN_SIZE` - Model hidden size
- `DEEP_USE_FRACTIONAL_DIFF` - Fractional differentiation flag
- `DEEP_FEATURE_SELECTION_METHOD` - Feature selection method

## Features

- Temporal Fusion Transformer (TFT)
- Quantile regression
- Feature selection
- Fractional differentiation
- Triple barrier labeling

## Related Documentation

- [Common Utilities](../common/) - DataFetcher, ExchangeManager
- [Config](../../modules/config.py) - Deep learning configuration

