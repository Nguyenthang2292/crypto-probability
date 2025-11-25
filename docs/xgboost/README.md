# XGBoost Prediction Documentation

Tài liệu cho XGBoost prediction component.

## Overview

XGBoost prediction component sử dụng machine learning (XGBoost) để dự đoán hướng di chuyển tiếp theo của cryptocurrency pairs.

## Components

### Model
- **Location:** `modules/xgboost/model.py`
- XGBoost model training và prediction
- Multi-class classification (UP, NEUTRAL, DOWN)

### Labeling
- **Location:** `modules/xgboost/labeling.py`
- Dynamic labeling dựa trên volatility
- Triple-barrier method với adaptive thresholds

### CLI
- **Location:** `modules/xgboost/cli.py`
- Command-line interface parser
- Input validation và prompts

### Display
- **Location:** `modules/xgboost/display.py`
- Classification report formatting
- Confusion matrix visualization

## Usage

```bash
python xgboost_prediction_main.py
```

## Configuration

Tất cả config được định nghĩa trong `modules/config.py` section **XGBoost Prediction Configuration**:
- `TARGET_HORIZON` - Số candles để predict ahead
- `TARGET_BASE_THRESHOLD` - Base threshold cho labeling
- `XGBOOST_PARAMS` - Model hyperparameters
- `MODEL_FEATURES` - List các features sử dụng

## Features

- Multi-exchange support với fallback
- Advanced technical indicators
- Dynamic threshold adjustment
- Real-time prediction với confidence scores

## Related Documentation

- [Common Utilities](../common/) - DataFetcher, ExchangeManager
- [Config](../../modules/config.py) - XGBoost configuration

