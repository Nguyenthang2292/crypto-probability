# Crypto Prediction AI / Dự Đoán Giá Cryptocurrency bằng AI

This project uses Machine Learning (XGBoost) to predict the next movement of cryptocurrency pairs.

Dự án này sử dụng Machine Learning (XGBoost) để dự đoán hướng di chuyển tiếp theo của các cặp tiền điện tử.

## Features / Tính Năng

-   **Multi-Exchange Support / Hỗ Trợ Đa Sàn**: Automatically fetches data from Binance, Kraken, KuCoin, Gate.io, etc. / Tự động lấy dữ liệu từ Binance, Kraken, KuCoin, Gate.io, v.v.
-   **Smart Fallback / Tự Động Chuyển Đổi**: If data is stale (e.g., delisted coin), it switches to another exchange. / Nếu dữ liệu cũ (ví dụ: coin bị delist), tự động chuyển sang sàn khác.
-   **Advanced Indicators / Chỉ Báo Nâng Cao**: Uses SMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, OBV, and Candlestick Patterns. / Sử dụng SMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, OBV, và các mẫu nến.
-   **Web UI / Giao Diện Web**: User-friendly Gradio interface for easy predictions. / Giao diện Gradio thân thiện, dễ sử dụng.
-   **Comprehensive Metrics / Đánh Giá Toàn Diện**: Classification reports with precision, recall, and F1-score for each direction. / Báo cáo phân loại với precision, recall, và F1-score cho từng hướng.

## Installation / Cài Đặt

1.  Install Python 3.8+. / Cài đặt Python 3.8 trở lên.
2.  Install dependencies: / Cài đặt các thư viện:
    ```bash
    pip install -r requirements.txt
    ```

## Usage / Cách Sử Dụng

### Option 1: Web UI (Recommended) / Tùy Chọn 1: Giao Diện Web (Khuyến Nghị)

Run the Gradio web interface: / Chạy giao diện web Gradio:
```bash
python crypto_ui_gradio.py
```

Then open your browser at `http://localhost:7860` / Sau đó mở trình duyệt tại `http://localhost:7860`

**Xem hướng dẫn chi tiết:** [README_UI.md](README_UI.md)

### Option 2: Command Line / Tùy Chọn 2: Dòng Lệnh

Run the CLI script: / Chạy script CLI:
```bash
python xgboost_prediction_main.py
```

Enter the symbol (e.g., `BTC/USDT`) and select the timeframe when prompted. / Nhập symbol (ví dụ: `BTC/USDT`) và chọn timeframe khi được yêu cầu.

### Option 3: Original Script / Tùy Chọn 3: Script Gốc

Run the original prediction script: / Chạy script dự đoán gốc:
```bash
python crypto_prediction.py
```

## Files / Các File

- `crypto_ui_gradio.py` - Web UI using Gradio (recommended) / Giao diện web sử dụng Gradio (khuyến nghị)
- `xgboost_prediction_main.py` - Enhanced CLI with advanced indicators / CLI nâng cao với các chỉ báo tiên tiến
- `crypto_prediction.py` - Original prediction script / Script dự đoán gốc
- `README_UI.md` - Detailed UI guide (Vietnamese) / Hướng dẫn UI chi tiết (Tiếng Việt)

## Documentation / Tài Liệu

- **English**: This README
- **Tiếng Việt**: [README_UI.md](README_UI.md) - Hướng dẫn chi tiết về giao diện web

## Disclaimer / Tuyên Bố Miễn Trừ

⚠️ **Not Financial Advice / Không Phải Lời Khuyên Tài Chính**: This tool is for educational purposes only. Trading cryptocurrency involves high risk. / Công cụ này chỉ dành cho mục đích giáo dục. Giao dịch tiền điện tử có rủi ro cao.

