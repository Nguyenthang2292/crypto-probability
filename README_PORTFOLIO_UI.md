# Portfolio Manager UI - Hướng dẫn sử dụng

## Tổng quan

`portfolio_ui.py` là giao diện web cho phép upload ảnh portfolio và tự động tách thông tin để phân tích.

## Tính năng

1. **Upload ảnh Portfolio**: Upload screenshot từ trading platform
2. **OCR tự động**: Tự động đọc text từ ảnh
3. **Parse thông tin**: Tách Symbol, Direction, Entry Price, Size
4. **Phân tích Portfolio**: Tính PnL, Delta, và đưa ra khuyến nghị
5. **Hedging Analysis**: Phân tích symbol mới để hedge

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cài đặt OCR Library

**Option 1: Pytesseract (Recommended)**
```bash
pip install pytesseract
```

Sau đó cài đặt Tesseract OCR:
- Windows: Download từ https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

**Option 2: EasyOCR (Alternative)**
```bash
pip install easyocr
```

## Chạy chương trình

```bash
python portfolio_ui.py
```

Chương trình sẽ mở web interface tại: `http://127.0.0.1:7860`

## Cách sử dụng

### Bước 1: Upload ảnh
1. Click vào "Upload Portfolio Image"
2. Chọn ảnh screenshot từ trading platform
3. Ảnh nên có format tương tự như ví dụ:
   - Có cột Symbol, Size, Entry Price
   - Text rõ ràng, không bị mờ

### Bước 2: Process Image
1. Click nút "Process Image"
2. Chương trình sẽ:
   - Đọc text từ ảnh (OCR)
   - Parse thông tin portfolio
   - Fetch giá hiện tại từ exchanges
   - Tính toán PnL và Delta

### Bước 3: Xem kết quả
- **Portfolio Analysis**: Hiển thị thống kê portfolio
- **Extracted Positions**: Bảng các vị thế đã tách được
- **Total PnL**: Tổng lãi/lỗ
- **Total Delta**: Tổng delta exposure

### Bước 4: Phân tích Hedging
1. Nhập symbol muốn phân tích (ví dụ: BTC/USDT)
2. Click "Analyze Symbol"
3. Xem khuyến nghị:
   - Direction (LONG/SHORT)
   - Size đề xuất
   - Correlation với portfolio

## Format ảnh yêu cầu

Ảnh nên có format tương tự:

```
Symbol          Size            Entry Price
DASHUSDT        -248.90        75.70
LSKUSDT         481.63         0.214454
LISTAUSDT       -255.99        0.2125000
...
```

**Lưu ý:**
- Text phải rõ ràng, không bị mờ
- Nên crop chỉ phần bảng portfolio
- Đảm bảo đủ ánh sáng, contrast tốt

## Troubleshooting

### Lỗi OCR không đọc được
1. Kiểm tra ảnh có rõ ràng không
2. Thử crop chỉ phần bảng
3. Tăng contrast của ảnh
4. Đảm bảo đã cài đặt Tesseract OCR

### Lỗi không parse được data
1. Kiểm tra OCR text output
2. Đảm bảo format ảnh đúng
3. Có thể cần chỉnh sửa regex patterns trong code

### Lỗi không fetch được giá
1. Kiểm tra kết nối internet
2. Symbol có thể không có trên exchanges
3. Chương trình sẽ tự động thử các exchanges khác

## Cải tiến có thể thêm

1. **Manual editing**: Cho phép chỉnh sửa data sau khi OCR
2. **Multiple image support**: Upload nhiều ảnh
3. **Export results**: Export kết quả ra CSV/Excel
4. **Better OCR**: Sử dụng AI models để OCR chính xác hơn
5. **Template matching**: Nhận diện format của từng trading platform

