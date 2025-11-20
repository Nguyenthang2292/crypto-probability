"""
UI for Portfolio Manager with Image Upload and OCR
"""
import gradio as gr
import numpy as np
import re
from portfolio_manager import PortfolioManager
from colorama import Fore, Style, init as colorama_init
from modules.utils import normalize_symbol

# Try to import image processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Initialize colorama
colorama_init(autoreset=True)

def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Style.RESET_ALL}"

def parse_portfolio_from_text(text: str):
    """
    Parse portfolio information from OCR text.
    Expected format: Symbol, Size, Entry Price
    """
    def parse_by_columns(raw_text: str):
        """Fallback parser for columnar OCR output (symbol column, size column, entry column)."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        lower_lines = [line.lower() for line in lines]
        try:
            symbol_idx = next(i for i, line in enumerate(lower_lines) if line == "symbol")
            size_idx = next(i for i, line in enumerate(lower_lines) if line == "size")
            entry_idx = next(i for i, line in enumerate(lower_lines) if "entry" in line and "price" in line)
        except StopIteration:
            return []

        symbol_lines = lines[symbol_idx + 1:size_idx]
        size_lines = lines[size_idx + 1:entry_idx]
        entry_lines = lines[entry_idx + 1:]

        symbols = []
        for raw in symbol_lines:
            if "perp" in raw.lower():
                continue
            match = re.search(r'([A-Z]{2,10})(?:/USDT|USDT)', raw.upper())
            if match:
                symbols.append(normalize_symbol(match.group(1)))

        def extract_numbers(raw_list):
            values = []
            for raw in raw_list:
                match = re.search(r'(-?\d[\d.,]*)', raw.replace(" ", ""))
                if match:
                    try:
                        values.append(float(match.group(1).replace(",", "")))
                    except ValueError:
                        continue
            return values

        sizes = extract_numbers(size_lines)
        entries = extract_numbers(entry_lines)

        count = min(len(symbols), len(sizes), len(entries))
        fallback_positions = []
        for i in range(count):
            size = sizes[i]
            fallback_positions.append({
                "symbol": symbols[i],
                "direction": "SHORT" if size < 0 else "LONG",
                "entry_price": entries[i],
                "size_usdt": abs(size),
            })

        return fallback_positions

    # Detect and parse columnar layouts first
    column_positions = parse_by_columns(text)
    if column_positions:
        return column_positions

    positions = []
    lines = text.strip().split('\n')
    
    # Common crypto symbols to help identify
    known_symbols = ['DASH', 'LSK', 'LISTA', 'DEGO', 'SKATE', 'WLFI', 'ENSO', 
                     'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE']
    
    current_symbol = None
    current_size = None
    current_entry = None
    
    for line in lines:
        line = line.strip()
        if not line:
            # Empty line might indicate end of a position, try to save it
            if current_symbol and current_size is not None and current_entry:
                direction = 'SHORT' if current_size < 0 else 'LONG'
                normalized_symbol = normalize_symbol(current_symbol)
                positions.append({
                    'symbol': normalized_symbol,
                    'direction': direction,
                    'entry_price': current_entry,
                    'size_usdt': abs(current_size)
                })
                current_symbol = None
                current_size = None
                current_entry = None
            continue
        
        # Remove common words that might confuse OCR
        line_clean = re.sub(r'perp\s*\d*x', '', line, flags=re.IGNORECASE)
        line_clean = re.sub(r'perpetual', '', line_clean, flags=re.IGNORECASE)
        line_clean = re.sub(r'\b\d+x\b', '', line_clean, flags=re.IGNORECASE) # Remove standalone leverage like "10x"
        line_clean = re.sub(r'P\d+\s*UT', '', line_clean, flags=re.IGNORECASE) # Remove "P10 UT" artifacts
        line_clean = re.sub(r'PE\d+X', '', line_clean, flags=re.IGNORECASE) # Remove "PE5X" artifacts
        line_clean = re.sub(r'USDT', ' ', line_clean, flags=re.IGNORECASE) # Replace USDT with space to separate numbers
        
        # Try to find symbol (usually contains known crypto names)
        for known in known_symbols:
            if known in line_clean.upper():
                # Extract full symbol (usually ends with USDT or similar)
                symbol_match = re.search(rf'({known}[A-Z]*)', line_clean.upper())
                if symbol_match:
                    if current_symbol and (current_size is not None or current_entry):
                         # Save previous
                         direction = 'SHORT' if current_size and current_size < 0 else 'LONG'
                         normalized_symbol = normalize_symbol(current_symbol)
                         positions.append({
                             'symbol': normalized_symbol,
                             'direction': direction,
                             'entry_price': current_entry if current_entry else 0.0,
                             'size_usdt': abs(current_size) if current_size else 0.0
                         })
                    
                    # Always clear values for the new symbol
                    current_size = None
                    current_entry = None
                    
                    current_symbol = normalize_symbol(symbol_match.group(1))
                    # Remove symbol from line so we don't parse it again or numbers inside it
                    line_clean = line_clean.replace(symbol_match.group(1), '')
                    break
        
        # Find numbers (could be size or entry price)
        numbers = re.findall(r'-?\d+\.?\d*', line_clean.replace(',', ''))
        

        for num_str in numbers:
            try:
                num = float(num_str)
                
                # Fix missing decimal for small prices read as integers
                # e.g. 05942176 (read as 5942176) -> 0.5942176
                # Only apply if we don't have a valid entry yet, or if the current entry is suspicious
                if num > 10000 and current_symbol not in ['BTC', 'YFI', 'PAXG', 'WBTC']:
                     # Try to normalize to 0.xxx or x.xxx
                     s_num = num_str.replace('.', '')
                     if s_num.startswith('0'):
                         # 05942176 -> 0.5942176
                         num = float("0." + s_num[1:])
                     elif len(s_num) >= 6:
                         # 5942176 -> 0.5942176 (assuming it was < 1)
                         num = float("0." + s_num)
                
                # Size is usually larger absolute value and can be negative
                # Entry price is usually positive and reasonable (0.0001 to 100000)
                if current_size is None:
                    # If negative or very large, likely size
                    if num < 0 or abs(num) > 100:
                        current_size = num
                        # Heuristic for missing decimal in size
                        if abs(current_size) > 10000:
                             s_size = str(int(abs(current_size)))
                             if len(s_size) > 4:
                                 # Assume 2 decimals were lost (common in USDT sizes)
                                 # 24861 -> 248.61
                                 current_size = current_size / 100.0
                                 
                    # If small positive number, might be entry price
                    elif 0.0001 <= num <= 100: # Increased range
                        current_entry = num
                elif current_entry is None:
                    # Second number is likely entry price
                    if 0.0001 <= num <= 100000:
                        current_entry = num
                    # If we have a size and this number is also large, maybe we swapped them?
                    # Or maybe this is the size and the previous was entry?
                    elif abs(num) > 100 and abs(current_size) < 100:
                        # Swap
                        current_entry = current_size
                        current_size = num
            except ValueError:
                continue
        
        # If we have all three, save the position
        if current_symbol and current_size is not None and current_entry:
            direction = 'SHORT' if current_size < 0 else 'LONG'
            normalized_symbol = normalize_symbol(current_symbol)
            positions.append({
                'symbol': normalized_symbol,
                'direction': direction,
                'entry_price': current_entry,
                'size_usdt': abs(current_size)
            })
            current_symbol = None
            current_size = None
            current_entry = None
    
    # Save last position if exists
    if current_symbol and current_size is not None and current_entry:
        direction = 'SHORT' if current_size < 0 else 'LONG'
        normalized_symbol = normalize_symbol(current_symbol)
        positions.append({
            'symbol': normalized_symbol,
            'direction': direction,
            'entry_price': current_entry,
            'size_usdt': abs(current_size)
        })
    
    return positions

def reconstruct_lines_from_data(data):
    """
    Reconstruct lines from Tesseract dict data based on Y-coordinates.
    This fixes issues where OCR reads down columns instead of across rows.
    """
    # Filter valid words
    words = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        # conf can be '-1' or string, cast to float/int safely
        try:
            conf = float(data['conf'][i])
        except (ValueError, TypeError):
            conf = -1
            
        # Accept all non-empty text, ignore confidence to catch faint text
        if data['text'][i].strip():
            words.append({
                'text': data['text'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
            
    if not words:
        return ""
        
    # Calculate average height to determine line tolerance
    avg_height = sum(w['height'] for w in words) / len(words)
    y_tolerance = avg_height * 0.5  # Stricter tolerance (50%) to avoid merging distinct lines
    
    # Sort by top
    words.sort(key=lambda x: x['top'])
    
    lines = []
    current_line = []
    # Initialize with first word
    if words:
        current_line = [words[0]]
        current_top = words[0]['top']
        
        for word in words[1:]:
            if word['top'] > current_top + y_tolerance:
                # New line
                if current_line:
                    current_line.sort(key=lambda x: x['left'])
                    lines.append(" ".join(w['text'] for w in current_line))
                current_line = [word]
                current_top = word['top']
            else:
                current_line.append(word)
                
        if current_line:
            current_line.sort(key=lambda x: x['left'])
            lines.append(" ".join(w['text'] for w in current_line))
        
    return "\n".join(lines)

def extract_text_from_image(image):
    """Extract text from image using OCR with advanced line reconstruction."""
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return "Error: Please install required libraries:\npip install opencv-python pillow"
    
    try:
        import pytesseract
        from pytesseract import Output
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(image)
        else:
            img_pil = image
        
        # Preprocess image for better OCR
        img_array = np.array(img_pil)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Try to get structured data first for better line reconstruction
        # Try multiple PSM modes
        for psm in [6, 4, 11, 12]:
            try:
                data = pytesseract.image_to_data(denoised, output_type=Output.DICT, config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/ ')
                text = reconstruct_lines_from_data(data)
                # Check if we found at least a few symbols
                if len(re.findall(r'[A-Z]{2,}/USDT', text)) >= 2:
                    return text
            except Exception as e:
                # print(f"Structured OCR failed for PSM {psm}: {e}")
                continue
        
        # Fallback to standard string extraction
        text = ""
        for psm in [6, 11, 12]:
            try:
                text = pytesseract.image_to_string(denoised, config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/ ')
                if len(text.strip()) > 20:
                    break
            except:
                continue
        
        return text if text else pytesseract.image_to_string(denoised, config='--psm 6')
        
    except ImportError:
        # Fallback to easyocr if pytesseract not available
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(np.array(image))
            # EasyOCR returns coordinates, we could use them too, but for now just join
            # To be consistent, we should sort EasyOCR results by Y too
            # results is list of (bbox, text, prob)
            # Sort by top-left Y
            results.sort(key=lambda x: x[0][0][1]) 
            # This is a simple sort, ideally we'd group lines like above
            text = '\n'.join([result[1] for result in results])
            return text
        except ImportError:
            return "Error: Please install OCR library.\nFor pytesseract: pip install pytesseract\nFor easyocr: pip install easyocr\n\nAlso install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
    except Exception as e:
        return f"OCR Error: {str(e)}\n\nPlease install OCR library:\npip install pytesseract easyocr"

def process_portfolio_image(image):
    """Process uploaded image and extract portfolio data."""
    if image is None:
        return "Please upload an image first.", None, None
    
    try:
        # Extract text from image
        ocr_text = extract_text_from_image(image)
        
        # Parse portfolio from text
        positions = parse_portfolio_from_text(ocr_text)
        
        if not positions:
            return f"Could not parse portfolio from image.\n\nOCR Text:\n{ocr_text}", None, None
        
        # Create portfolio manager and add positions
        pm = PortfolioManager()
        for pos in positions:
            pm.add_position(
                pos['symbol'],
                pos['direction'],
                pos['entry_price'],
                pos['size_usdt']
            )
        
        # Fetch prices
        pm.fetch_prices()
        
        # Calculate stats
        df, total_pnl, total_delta = pm.calculate_stats()
        
        # Format output
        output_text = f"=== PORTFOLIO EXTRACTED ===\n\n"
        output_text += f"Found {len(positions)} positions:\n\n"
        
        for pos in positions:
            output_text += f"{pos['symbol']:15} {pos['direction']:5} Entry: {pos['entry_price']:>12.8f} Size: {pos['size_usdt']:>12.2f} USDT\n"
        
        output_text += f"\n=== PORTFOLIO STATISTICS ===\n"
        if not df.empty:
            output_text += df.to_string(index=False)
            output_text += f"\n\nTotal PnL: {total_pnl:.2f} USDT\n"
        output_text += f"Total Delta: {total_delta:.2f} USDT\n"
        
        # Analysis
        if total_delta > 0:
            output_text += f"\nPortfolio has LONG exposure. Consider SHORT positions to hedge."
        elif total_delta < 0:
            output_text += f"\nPortfolio has SHORT exposure. Consider LONG positions to hedge."
        else:
            output_text += f"\nPortfolio is Delta Neutral."
        
        # Create table for display
        table_data = []
        for pos in positions:
            table_data.append([
                pos['symbol'],
                pos['direction'],
                f"{pos['entry_price']:.8f}",
                f"{pos['size_usdt']:.2f}"
            ])
        
        return output_text, table_data, pm
        
    except Exception as e:
        return f"Error processing image: {str(e)}\n\nPlease check if OCR libraries are installed:\npip install pytesseract easyocr", None, None

def analyze_symbol(pm, symbol, total_delta, correlation_mode_label):
    """Analyze a symbol for hedging."""
    if pm is None or not symbol:
        return "Please process portfolio image first."
    
    try:
        mode_key = (
            "portfolio_return"
            if correlation_mode_label == "Portfolio Return Correlation"
            else "weighted"
        )
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(
            symbol, total_delta, correlation_mode=mode_key
        )
        
        result = f"=== ANALYSIS FOR {symbol} ===\n\n"
        result += f"Correlation Mode: {correlation_mode_label}\n"
        if recommended_direction:
            result += f"Recommended Direction: {recommended_direction}\n"
            result += f"Recommended Size: {recommended_size:.2f} USDT\n"
            if correlation is not None:
                corr_status = "High" if abs(correlation) > 0.7 else ("Moderate" if abs(correlation) > 0.4 else "Low")
                result += f"Correlation: {correlation:.4f} ({corr_status})\n"
        else:
            result += "Portfolio is already delta neutral.\n"
        
        return result
    except Exception as e:
        return f"Error analyzing symbol: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Portfolio Manager - Image Upload") as demo:
        gr.Markdown("# 📊 Portfolio Manager - Image Upload & Analysis")
        gr.Markdown("Upload an image of your portfolio positions and get automatic analysis.")
        
        with gr.Tabs():
            with gr.Tab("📷 Image Upload (OCR)"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload or Paste Portfolio Image",
                            type="numpy",
                            sources=["upload", "clipboard", "webcam"],
                            height=450,
                        )
                        process_btn = gr.Button("Process Image", variant="primary")
                        gr.Markdown("""
                        **Instructions:**
                        - Upload or paste (Ctrl+V) a screenshot of your portfolio
                        - Ensure text is clear and readable
                        - Image should show Symbol, Size, and Entry Price columns
                        """)
                    
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="Portfolio Analysis",
                            lines=20,
                            max_lines=30
                        )
            
            with gr.Tab("✏️ Manual Input"):
                gr.Markdown("### Enter Portfolio Positions Manually")
                manual_input = gr.Textbox(
                    label="Portfolio Data",
                    placeholder="""Enter positions in format:
                        DASH/USDT SHORT 75.70 248.90
                        LSK/USDT LONG 0.214454 481.63
                        LISTA/USDT SHORT 0.2125000 255.99
                        ...
                        Format: SYMBOL DIRECTION ENTRY_PRICE SIZE_USDT""",
                    lines=10
                )
                manual_process_btn = gr.Button("Process Manual Input", variant="primary")
        
        portfolio_table = gr.Dataframe(
            label="Extracted Positions",
            headers=["Symbol", "Direction", "Entry Price", "Size (USDT)"],
            interactive=False
        )
        
        with gr.Row():
            symbol_input = gr.Textbox(
                label="Symbol to Analyze (e.g., BTC/USDT)",
                placeholder="Enter symbol for hedging analysis"
            )
            correlation_mode = gr.Radio(
                label="Correlation Mode",
                choices=[
                    "Weighted Correlation (by Position Size)",
                    "Portfolio Return Correlation",
                ],
                value="Weighted Correlation (by Position Size)",
            )
            analyze_btn = gr.Button("Analyze Symbol", variant="secondary")
        
        analysis_output = gr.Textbox(
            label="Hedging Analysis",
            lines=10
        )
        
        # Store portfolio manager in state
        pm_state = gr.State(value=None)
        delta_state = gr.State(value=0.0)
        
        def process_image(image):
            text, table, pm = process_portfolio_image(image)
            total_delta = 0.0
            if pm:
                _, _, total_delta = pm.calculate_stats()
            return text, table, pm, total_delta
        
        def process_manual_input(text_input):
            """Process manual text input."""
            if not text_input:
                return "Please enter portfolio data.", None, None, 0.0
            
            try:
                positions = []
                lines = text_input.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    
                    try:
                        symbol = normalize_symbol(parts[0])
                        direction = parts[1].upper()
                        entry_price = float(parts[2])
                        size_usdt = float(parts[3])
                        
                        positions.append({
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': entry_price,
                            'size_usdt': abs(size_usdt)
                        })
                    except (ValueError, IndexError):
                        continue
                
                if not positions:
                    return "Could not parse any positions. Please check format.", None, None, 0.0
                
                # Create portfolio manager
                pm = PortfolioManager()
                for pos in positions:
                    pm.add_position(
                        pos['symbol'],
                        pos['direction'],
                        pos['entry_price'],
                        pos['size_usdt']
                    )
                
                # Fetch prices
                pm.fetch_prices()
                
                # Calculate stats
                df, total_pnl, total_delta = pm.calculate_stats()
                
                # Format output
                output_text = f"=== PORTFOLIO LOADED ===\n\n"
                output_text += f"Loaded {len(positions)} positions:\n\n"
                
                for pos in positions:
                    output_text += f"{pos['symbol']:15} {pos['direction']:5} Entry: {pos['entry_price']:>12.8f} Size: {pos['size_usdt']:>12.2f} USDT\n"
                
                output_text += f"\n=== PORTFOLIO STATISTICS ===\n"
                if not df.empty:
                    output_text += df.to_string(index=False)
                    output_text += f"\n\nTotal PnL: {total_pnl:.2f} USDT\n"
                output_text += f"Total Delta: {total_delta:.2f} USDT\n"
                
                if total_delta > 0:
                    output_text += f"\nPortfolio has LONG exposure. Consider SHORT positions to hedge."
                elif total_delta < 0:
                    output_text += f"\nPortfolio has SHORT exposure. Consider LONG positions to hedge."
                else:
                    output_text += f"\nPortfolio is Delta Neutral."
                
                # Create table
                table_data = []
                for pos in positions:
                    table_data.append([
                        pos['symbol'],
                        pos['direction'],
                        f"{pos['entry_price']:.8f}",
                        f"{pos['size_usdt']:.2f}"
                    ])
                
                return output_text, table_data, pm, total_delta
                
            except Exception as e:
                return f"Error processing manual input: {str(e)}", None, None, 0.0
        
        def analyze_with_pm(pm, delta, symbol, correlation_mode_label):
            if pm is None:
                return "Please process portfolio first."
            return analyze_symbol(pm, symbol, delta, correlation_mode_label)
        
        process_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[output_text, portfolio_table, pm_state, delta_state]
        )
        
        manual_process_btn.click(
            fn=process_manual_input,
            inputs=[manual_input],
            outputs=[output_text, portfolio_table, pm_state, delta_state]
        )
        
        analyze_btn.click(
            fn=analyze_with_pm,
            inputs=[pm_state, delta_state, symbol_input, correlation_mode],
            outputs=[analysis_output]
        )
        
        gr.Markdown("""
        ### Instructions:
        1. Upload an image of your portfolio positions (screenshot from trading platform)
        2. Click "Process Image" to extract positions
        3. Review the extracted portfolio data
        4. Enter a symbol to analyze for hedging opportunities
        5. Click "Analyze Symbol" to get recommendations
        
        ### Requirements:
        - Install OCR library: `pip install pytesseract` or `pip install easyocr`
        - For pytesseract, also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)

