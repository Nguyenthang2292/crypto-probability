# Decision Matrix Implementation - Hướng dẫn sử dụng

## Tổng quan

Đã triển khai 2 phương án tích hợp Decision Matrix vào ATC + Range Oscillator + SPC workflow:

1. **Phương án 1 (Hybrid)**: `main_atc_oscillator_spc_hybrid.py`
   - Kết hợp sequential filtering và voting system
   - Workflow: ATC → Range Oscillator → SPC → Decision Matrix Voting

2. **Phương án 2 (Pure Voting)**: `main_atc_oscillator_spc_voting.py`
   - Thay thế hoàn toàn bằng voting system
   - Workflow: Calculate all signals → Voting System → Final Results

## Files đã tạo

### Main Files
- `main_atc_oscillator_spc_hybrid.py` - Phương án 1: Hybrid Approach
- `main_atc_oscillator_spc_voting.py` - Phương án 2: Pure Voting System

### Module Files
- `modules/decision_matrix/__init__.py` - Module init
- `modules/decision_matrix/classifier.py` - DecisionMatrixClassifier class

## Cài đặt

Không cần cài đặt thêm, chỉ cần đảm bảo các dependencies hiện có đã được cài đặt.

## Sử dụng

### Phương án 1: Hybrid Approach

**Workflow:**
1. ATC Scan → tìm signals ban đầu
2. Range Oscillator Filter → xác nhận signals
3. SPC Filter (optional) → xác nhận thêm
4. Decision Matrix Voting (optional) → voting system
5. Final Results

**Ví dụ chạy:**

```bash
# Chạy với SPC và Decision Matrix
python main_atc_oscillator_spc_hybrid.py \
    --timeframe 1h \
    --enable-spc \
    --spc-strategy cluster_transition \
    --use-decision-matrix \
    --voting-threshold 0.6 \
    --min-votes 2

# Chạy chỉ với SPC (không dùng Decision Matrix)
python main_atc_oscillator_spc_hybrid.py \
    --timeframe 1h \
    --enable-spc \
    --spc-strategy regime_following

# Chạy không có SPC (giống main_atc_oscillator.py cũ)
python main_atc_oscillator_spc_hybrid.py --timeframe 1h
```

**Command-line Arguments:**

**SPC Options:**
- `--enable-spc`: Bật SPC filtering
- `--spc-strategy`: Chọn strategy (cluster_transition, regime_following, mean_reversion)
- `--spc-k`: Số clusters (2 hoặc 3)
- `--spc-lookback`: Số bars lịch sử
- `--spc-p-low`, `--spc-p-high`: Percentiles
- `--spc-min-signal-strength`, `--spc-min-rel-pos-change`: Cluster transition params
- `--spc-min-regime-strength`, `--spc-min-cluster-duration`: Regime following params
- `--spc-extreme-threshold`, `--spc-min-extreme-duration`: Mean reversion params

**Decision Matrix Options:**
- `--use-decision-matrix`: Bật voting system
- `--voting-threshold`: Ngưỡng weighted score (default: 0.5)
- `--min-votes`: Số indicators tối thiểu phải agree (default: 2)

### Phương án 2: Pure Voting System

**Workflow:**
1. ATC Scan → tìm signals ban đầu
2. Calculate all signals → tính signals từ tất cả indicators song song
3. Voting System → áp dụng voting system
4. Final Results

**Ví dụ chạy:**

```bash
# Chạy với SPC
python main_atc_oscillator_spc_voting.py \
    --timeframe 1h \
    --enable-spc \
    --spc-strategy cluster_transition \
    --voting-threshold 0.6 \
    --min-votes 2

# Chạy không có SPC (chỉ ATC + Range Oscillator)
python main_atc_oscillator_spc_voting.py \
    --timeframe 1h \
    --voting-threshold 0.5 \
    --min-votes 2
```

**Command-line Arguments:**

Tương tự như Phương án 1, nhưng:
- **Không có** `--use-decision-matrix` (voting system luôn được dùng)
- `--voting-threshold` và `--min-votes` là bắt buộc

## So sánh 2 phương án

### Phương án 1: Hybrid Approach

**Ưu điểm:**
- ✅ Linh hoạt: Có thể bật/tắt từng bước
- ✅ Backward compatible: Có thể chạy như workflow cũ
- ✅ Có fallback logic: Tự động fallback nếu không có signals match
- ✅ Có thể so sánh sequential vs voting

**Nhược điểm:**
- ⚠️ Phức tạp hơn: Nhiều bước hơn
- ⚠️ Có thể chậm hơn: Sequential filtering

**Khi nào dùng:**
- Khi muốn giữ logic cũ và thêm voting system
- Khi muốn so sánh 2 approaches
- Khi muốn có fallback logic

### Phương án 2: Pure Voting System

**Ưu điểm:**
- ✅ Đơn giản hơn: Chỉ có voting system
- ✅ Nhanh hơn: Tính signals song song
- ✅ Dễ mở rộng: Dễ thêm indicators mới
- ✅ Có đầy đủ metrics: Feature importance, accuracy, weighted impact

**Nhược điểm:**
- ⚠️ Thay đổi lớn: Khác hoàn toàn workflow cũ
- ⚠️ Không có fallback: Phải dựa vào voting

**Khi nào dùng:**
- Khi muốn approach mới hoàn toàn
- Khi muốn đơn giản hóa workflow
- Khi muốn tính signals song song

## Output

Cả 2 phương án đều hiển thị:

1. **Final Results**: Danh sách symbols với confirmed signals
2. **Voting Metadata** (nếu dùng Decision Matrix):
   - Weighted Score
   - Voting Breakdown (Node 1, Node 2, Node 3)
   - Feature Importance
   - Weighted Impact
   - Independent Accuracy

**Ví dụ output:**

```
LONG Signals - Voting Breakdown:
--------------------------------------------------------------------------------

Symbol: BTC/USDT
  Weighted Score: 72.50%
  Voting Breakdown:
    ATC: ✓ (Weight: 33.3%, Impact: 33.3%, Importance: 65.0%, Contribution: 0.22)
    OSCILLATOR: ✓ (Weight: 33.3%, Impact: 33.3%, Importance: 70.0%, Contribution: 0.23)
    SPC: ✓ (Weight: 33.3%, Impact: 33.3%, Importance: 68.0%, Contribution: 0.23)
```

## Performance

### Phương án 1 (Hybrid)
- Sequential filtering: Có thể chậm hơn
- Parallel processing trong mỗi bước
- Tổng thời gian: ~3-5 phút cho 100 symbols

### Phương án 2 (Pure Voting)
- Parallel calculation: Nhanh hơn
- Tính tất cả signals một lần
- Tổng thời gian: ~2-3 phút cho 100 symbols

## Troubleshooting

### Lỗi: "No ATC signals found"
- Kiểm tra timeframe và limit
- Kiểm tra kết nối exchange

### Lỗi: "No signals confirmed"
- Giảm `--voting-threshold`
- Giảm `--min-votes`
- Kiểm tra SPC parameters

### Lỗi: Import errors
- Đảm bảo đã cài đặt tất cả dependencies
- Kiểm tra Python path

## Next Steps

1. **Backtesting**: So sánh performance của 2 phương án
2. **Tuning**: Điều chỉnh voting threshold và min_votes
3. **Accuracy Tracking**: Tính toán accuracy thực tế từ historical data
4. **Feature Selection**: Tự động chọn indicators tốt nhất

## Tham khảo

- `DECISION_MATRIX_ANALYSIS.md`: Phân tích chi tiết
- `DECISION_MATRIX_IMPLEMENTATION.py`: Code example
- `TOM_TAT_DECISION_MATRIX.md`: Tóm tắt bằng tiếng Việt
- `Document1.pdf`: Tài liệu gốc về Decision Matrix

