# Crypto Prediction UI - User Guide / Hướng dẫn sử dụng (Gradio)

## 🚀 Quick Start / Bắt Đầu Nhanh

### Quick Installation / Cài đặt nhanh

```bash
# 1. Install dependencies / Cài đặt thư viện
pip install -r requirements.txt

# 2. Run UI / Chạy UI
python crypto_ui_gradio.py
```

### Quick Usage / Sử dụng nhanh

1. Open your browser at `http://localhost:7860` (or the address shown in terminal) / Mở trình duyệt tại `http://localhost:7860` (hoặc địa chỉ hiển thị trong terminal)
2. Fill in the form on the left: / Điền thông tin ở form bên trái:
   - **Trading Pair**: `BTC` or `ETH` / `BTC` hoặc `ETH`
   - **Timeframe**: `1h` (recommended) / `1h` (khuyến nghị)
   - **Number of Candles**: `1500` (recommended) / `1500` (khuyến nghị)
   - **Exchanges**: Check the exchanges (all selected by default) / Tích chọn các sàn (mặc định đã chọn tất cả)
3. Click the **"🚀 Predict"** button (blue, large) / Click nút **"🚀 Predict"** (màu xanh, lớn)
4. Wait for results to appear on the right column! / Đợi kết quả hiển thị ở cột bên phải!

---

## 📋 Requirements / Yêu cầu

### Install Libraries / Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Or install manually: / Hoặc cài đặt thủ công:
```bash
pip install gradio plotly
```

## 🚀 Running the Application / Chạy ứng dụng

### Launch UI / Khởi động UI

```bash
python crypto_ui_gradio.py
```

The application will display the access address in the terminal. Open your browser at: / Ứng dụng sẽ hiển thị địa chỉ truy cập trong terminal. Mở trình duyệt tại:
- `http://localhost:7860` or / hoặc
- `http://127.0.0.1:7860`

## 📖 Detailed Usage Guide / Hướng dẫn sử dụng chi tiết

### 1. Configure Parameters / Cấu hình tham số

**Left Column (Configuration): / Cột bên trái (Configuration):**

- **Trading Pair**: Enter symbol (e.g., `BTC`, `ETH`) or full pair (`BTC/USDT`) / Nhập symbol (ví dụ: `BTC`, `ETH`) hoặc cặp đầy đủ (`BTC/USDT`)
- **Quote Currency**: Select quote currency from dropdown (USDT, USD, BTC, ETH) / Chọn đồng quote từ dropdown (USDT, USD, BTC, ETH)
- **Timeframe**: Select timeframe from dropdown (30m, 45m, 1h, 2h, 4h, 6h, 12h, 1d) / Chọn khung thời gian từ dropdown (30m, 45m, 1h, 2h, 4h, 6h, 12h, 1d)
- **Number of Candles**: Drag slider to select number of candles (500-3000) / Kéo slider để chọn số lượng nến (500-3000)
  - More = more training data but slower / Nhiều hơn = nhiều dữ liệu huấn luyện hơn nhưng chậm hơn
  - Recommended: 1500-2000 / Khuyến nghị: 1500-2000
- **Exchanges**: Check exchanges to fetch data from / Tích chọn các sàn giao dịch để lấy dữ liệu
  - Should select multiple exchanges for better reliability / Nên chọn nhiều sàn để đảm bảo độ tin cậy

### 2. Make Prediction / Thực hiện dự đoán

1. Fill in the form on the left / Điền thông tin vào form bên trái
2. Click the **"🚀 Predict"** button (blue, large) / Click nút **"🚀 Predict"** (màu xanh, lớn)
3. Wait for the process (may take a few minutes): / Đợi quá trình (có thể mất vài phút):
   - Fetching data (Getting data from exchanges) / Lấy dữ liệu từ exchanges
   - Calculating indicators (Computing technical indicators) / Tính toán chỉ báo kỹ thuật
   - Training model (Training XGBoost model) / Huấn luyện mô hình XGBoost
   - Making prediction (Generating prediction) / Đưa ra dự đoán
4. View results in the right column / Xem kết quả ở cột bên phải

### 3. Read Results / Đọc kết quả

**Main Results: / Kết quả chính:**
- **Prediction**: Predicted direction (UP/DOWN/NEUTRAL) / Hướng dự đoán (UP/DOWN/NEUTRAL)
- **Confidence**: Confidence level (%) / Độ tin cậy (%)

**Additional Information: / Thông tin bổ sung:**
- **Status**: Current status (Success/Error) / Trạng thái hiện tại (Success/Error)
- **Prediction Results**: Detailed prediction results with markdown formatting / Kết quả dự đoán chi tiết với markdown formatting
- **Price Chart**: Interactive candlestick chart (tab "📈 Price Chart") / Biểu đồ nến tương tác (tab "📈 Price Chart")
- **Probability Chart**: Bar chart showing probabilities (tab "📊 Probability Chart") / Biểu đồ cột hiển thị xác suất (tab "📊 Probability Chart")
- **Technical Indicators**: All technical indicators displayed in results / Tất cả chỉ báo kỹ thuật được hiển thị trong kết quả
- **Price Targets**: Price targets based on ATR multiples (if not NEUTRAL) / Mục tiêu giá dựa trên ATR multiples (nếu không phải NEUTRAL)

## 🎨 UI Features / Tính năng UI

### Gradio Interface / Giao diện Gradio

- **2-Column Layout**: Input form on left, results on right / Layout 2 cột: Form input bên trái, kết quả bên phải
- **Tabs**: Switch between Price Chart and Probability Chart / Chuyển đổi giữa Price Chart và Probability Chart
- **Real-time Updates**: Status and results update immediately / Status và kết quả cập nhật ngay khi có

### Interactive Charts (Plotly) / Biểu đồ tương tác (Plotly)

- **Price Chart**: Candlestick chart with volume / Biểu đồ nến (candlestick) với volume
- **Probability Chart**: Bar chart showing probabilities for UP/NEUTRAL/DOWN / Biểu đồ cột hiển thị xác suất cho UP/NEUTRAL/DOWN

### Prediction Colors / Màu sắc dự đoán

- 🟢 **UP**: Green color (#28a745) / Màu xanh lá (#28a745)
- 🔴 **DOWN**: Red color (#dc3545) / Màu đỏ (#dc3545)
- 🟡 **NEUTRAL**: Yellow color (#ffc107) / Màu vàng (#ffc107)

### Detailed Information / Thông tin chi tiết

- Markdown formatting for readable results / Markdown formatting cho kết quả dễ đọc
- All technical indicators displayed / Tất cả technical indicators được hiển thị
- Error handling with clear error messages / Error handling với thông báo lỗi rõ ràng

## 💡 Usage Tips / Mẹo sử dụng

1. **For Best Results: / Để có kết quả tốt nhất:**
   - Use at least 1500 candles / Sử dụng ít nhất 1500 candles
   - Select multiple exchanges / Chọn nhiều exchanges
   - Timeframe 1h or 4h usually gives good results / Timeframe 1h hoặc 4h thường cho kết quả tốt

2. **Understand Predictions: / Hiểu rõ dự đoán:**
   - Model predicts for the next **24 candles** / Model dự đoán cho **24 candles** tiếp theo
   - Dynamic threshold based on historical volatility / Threshold động dựa trên biến động lịch sử
   - Precision of UP/DOWN is more important than overall accuracy / Precision của UP/DOWN quan trọng hơn accuracy tổng thể

3. **Error Handling: / Xử lý lỗi:**
   - If data fetch fails: Try again or select different exchange / Nếu không lấy được dữ liệu: Thử lại hoặc chọn exchange khác
   - If training is slow: Reduce number of candles / Nếu training chậm: Giảm số lượng candles
   - If insufficient data: Increase limit / Nếu không đủ dữ liệu: Tăng limit

## 🔧 Troubleshooting / Xử Lý Sự Cố

### Gradio Import Error / Lỗi import gradio

```bash
# Make sure all dependencies are installed / Đảm bảo đã cài đặt đầy đủ
pip install -r requirements.txt
# or / hoặc
pip install gradio plotly
```

### Exchange Connection Error / Lỗi kết nối exchange

- Check internet connection / Kiểm tra kết nối internet
- Try selecting different exchange (uncheck some exchanges) / Thử chọn exchange khác (bỏ tích một số exchange)
- Some exchanges may be blocked in some countries / Một số exchange có thể bị chặn ở một số quốc gia

### UI Not Displaying or Cannot Access / UI không hiển thị hoặc không truy cập được

- Check terminal for errors / Kiểm tra terminal có lỗi không
- Make sure to access the correct address: `http://localhost:7860` (not `0.0.0.0`) / Đảm bảo truy cập đúng địa chỉ: `http://localhost:7860` (không phải `0.0.0.0`)
- Try refreshing browser: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac) / Thử refresh trình duyệt: `Ctrl + Shift + R` (Windows) hoặc `Cmd + Shift + R` (Mac)
- Check if port 7860 is occupied: `netstat -ano | findstr :7860` (Windows) / Kiểm tra port 7860 có bị chiếm không: `netstat -ano | findstr :7860` (Windows)

### Prediction Error / Lỗi khi predict

- Check if symbol is correct (e.g., BTC, ETH, not BTCUSDT) / Kiểm tra symbol có đúng không (ví dụ: BTC, ETH, không phải BTCUSDT)
- Make sure at least 1 exchange is selected / Đảm bảo đã chọn ít nhất 1 exchange
- Check Error Details at the bottom of the page if there's an error / Xem Error Details ở cuối trang nếu có lỗi

### Port Already in Use / Port đã được sử dụng

- Close other applications using port 7860 / Đóng ứng dụng khác đang dùng port 7860
- Or change port in code: `server_port=7861` / Hoặc sửa port trong code: `server_port=7861`

## 📝 Notes / Lưu ý

- **Not Financial Advice / Không phải lời khuyên đầu tư**: This is an analysis tool, not financial advice / Đây là công cụ phân tích, không phải lời khuyên tài chính
- **Risk / Rủi ro**: Trading cryptocurrency involves high risk, only invest what you can afford to lose / Trading cryptocurrency có rủi ro cao, chỉ đầu tư số tiền bạn có thể mất
- **Backtesting / Backtesting**: Always backtest before using in real trading / Luôn backtest trước khi sử dụng trong thực tế
- **Model Accuracy / Model accuracy**: Models can be wrong, always combine with other technical analysis / Model có thể sai, luôn kết hợp với phân tích kỹ thuật khác
- **First Run / Lần đầu chạy**: May take a few minutes to train the model / Có thể mất vài phút để train model
- **Internet Connection / Kết nối internet**: Internet connection required to fetch data from exchanges / Cần kết nối internet để fetch data từ exchanges

## 🆚 Comparison with CLI / So sánh với CLI

| Feature / Tính năng | CLI (`xgboost_prediction_main.py`) | UI (`crypto_ui_gradio.py`) |
|---------------------|----------------------------------|---------------------------|
| Ease of Use / Dễ sử dụng | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed / Tốc độ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Visualization / Visualization | ❌ | ✅ |
| Interactive / Interactive | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Automation / Automation | ✅ | ⭐⭐ |
| Metrics Detail / Metrics detail | ✅ | ⭐⭐⭐⭐ |
| No ScriptRunContext Error / Không lỗi ScriptRunContext | ✅ | ✅ |
| Interactive Charts / Charts tương tác | ❌ | ✅ (Plotly) |

## 🔗 Links / Liên kết

- Main File / File chính: `xgboost_prediction_main.py`
- UI File / File UI: `crypto_ui_gradio.py`
- Requirements: `requirements.txt`
- Main README: [README.md](README.md)
