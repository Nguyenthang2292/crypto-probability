# 📚 Documentation

Thư mục này chứa tất cả các file tài liệu (.md) của project, được tổ chức theo components để dễ quản lý và tìm kiếm.

## Cấu Trúc

```text
docs/
├── README.md                    # File này - Tổng quan documentation
│
├── common/                      # Shared utilities documentation
│   ├── README.md
│   └── ExchangeManager.md
│
├── xgboost/                     # XGBoost prediction documentation
│   ├── README.md
│   └── TARGET_HORIZON_EXPLANATION.md
│
├── portfolio/                   # Portfolio management documentation
│   └── README.md
│
├── deeplearning/                # Deep learning documentation
│   ├── README.md
│   ├── deeplearning_data_pipeline.md
│   ├── deeplearning_model.md
│   ├── deep_prediction_training.md
│   └── feature_selection.md
│
├── hmm/                         # HMM signal generation documentation
│   └── (tài liệu sẽ được thêm sau)
│
└── pairs_trading/               # Pairs trading documentation
    └── README.md
```

## Components

### 🔧 Common / Shared Utilities

Tài liệu cho các modules dùng chung cho tất cả components:
- **[ExchangeManager](./common/ExchangeManager.md)** - Quản lý kết nối với các exchanges
- Xem [README](./common/README.md) để biết thêm chi tiết

### 📊 XGBoost Prediction

Tài liệu cho XGBoost prediction component:
- **[Target Horizon Explanation](./xgboost/TARGET_HORIZON_EXPLANATION.md)** - Giải thích về target horizon và prediction windows
- Xem [README](./xgboost/README.md) để biết thêm chi tiết

### 💼 Portfolio Manager

Tài liệu cho portfolio management component:
- Xem [README](./portfolio/README.md) để biết thêm chi tiết về:
  - PortfolioCorrelationAnalyzer - Phân tích correlation giữa portfolio và symbols
  - Risk Calculator - Tính toán PnL, Delta, Beta, VaR
  - Hedge Finder - Tìm hedge candidates

### 🧠 Deep Learning

Tài liệu cho deep learning prediction component:
- **[Data Pipeline](./deeplearning/deeplearning_data_pipeline.md)** - Pipeline chuẩn bị data cho TFT
- **[Model](./deeplearning/deeplearning_model.md)** - TFT model architecture
- **[Training](./deeplearning/deep_prediction_training.md)** - Hướng dẫn training
- **[Feature Selection](./deeplearning/feature_selection.md)** - Chọn lọc và kỹ thuật hóa features
- Xem [README](./deeplearning/README.md) để biết thêm chi tiết

### 🔄 Pairs Trading

Tài liệu cho pairs trading analysis component:
- Xem [README](./pairs_trading/README.md) để biết thêm chi tiết

### 📈 HMM Signal Generation

Tài liệu cho HMM (Hidden Markov Model) signal generation component:
- High-Order HMM và HMM-KAMA models để tạo trading signals
- Signal combining, confidence scoring, và conflict resolution
- (Tài liệu chi tiết sẽ được thêm sau)

## Lưu Ý

- Tất cả các file documentation (.md) được tổ chức theo components
- Các link nội bộ giữa các file .md sử dụng relative path
- Mỗi component có README.md riêng để mô tả chi tiết
- Không nên đặt file .md trong thư mục `modules/` để tránh lẫn với code

## Quick Links

- [Common Utilities](./common/)
- [XGBoost Prediction](./xgboost/)
- [Portfolio Manager](./portfolio/)
- [Deep Learning](./deeplearning/)
- [HMM Signal Generation](./hmm/)
- [Pairs Trading](./pairs_trading/)
