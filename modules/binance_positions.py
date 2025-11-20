"""
Binance Futures USDT-M Position Fetcher
Lấy thông tin các position đang mở từ Binance Futures USDT-M
"""
import ccxt
import os
from typing import List, Dict
from colorama import Fore, Style, init as colorama_init

try:
    from .config_api import BINANCE_API_KEY, BINANCE_API_SECRET
except ImportError:
    # Fallback nếu file config_api.py chưa tồn tại
    BINANCE_API_KEY = None
    BINANCE_API_SECRET = None

# Initialize colorama
colorama_init(autoreset=True)

def get_binance_futures_positions(api_key: str = None, api_secret: str = None, testnet: bool = False, debug: bool = False) -> List[Dict]:
    """
    Lấy danh sách các position đang mở từ Binance Futures USDT-M
    
    Args:
        api_key: API Key từ Binance. Thứ tự ưu tiên:
            1. Tham số này (nếu được cung cấp)
            2. Biến môi trường BINANCE_API_KEY
            3. modules.config_api.BINANCE_API_KEY
        api_secret: API Secret từ Binance. Thứ tự ưu tiên:
            1. Tham số này (nếu được cung cấp)
            2. Biến môi trường BINANCE_API_SECRET
            3. modules.config_api.BINANCE_API_SECRET
        testnet: Sử dụng testnet nếu True (mặc định: False)
        debug: Hiển thị debug info nếu True (mặc định: False)
    
    Returns:
        List các dictionary chứa thông tin position
    """
    # Lấy API key và secret theo thứ tự ưu tiên:
    # 1. Tham số hàm
    # 2. Biến môi trường
    # 3. Config từ modules.config_api
    if api_key is None:
        api_key = os.getenv('BINANCE_API_KEY') or BINANCE_API_KEY
    if api_secret is None:
        api_secret = os.getenv('BINANCE_API_SECRET') or BINANCE_API_SECRET
    
    if not api_key or not api_secret:
        raise ValueError(
            "API Key và API Secret là bắt buộc!\n"
            "Cung cấp qua một trong các cách sau:\n"
            "  1. Tham số command line: --api-key và --api-secret\n"
            "  2. Biến môi trường: BINANCE_API_KEY và BINANCE_API_SECRET\n"
            "  3. File config: modules/config_api.py (BINANCE_API_KEY và BINANCE_API_SECRET)"
        )
    
    # Khởi tạo exchange
    options = {
        'defaultType': 'future',  # Sử dụng Futures
        'options': {
            'defaultType': 'future',
        }
    }
    
    if testnet:
        # Sử dụng Binance Testnet
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': options,
            'sandbox': True,  # Testnet mode
        })
        print(f"{Fore.YELLOW}⚠️  Đang sử dụng Binance Testnet{Style.RESET_ALL}")
    else:
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': options,
        })
    
    try:
        # Lấy tất cả positions
        positions = exchange.fetch_positions()
        
        # Lọc chỉ các position đang mở (size != 0) và là USDT-M
        open_positions = []
        for pos in positions:
            # Lấy thông tin contracts từ nhiều nguồn có thể
            # ccxt có thể trả về trong 'contracts' hoặc 'positionAmt'
            contracts = float(pos.get('contracts', 0) or 0)
            if contracts == 0:
                # Thử lấy từ positionAmt
                position_amt = pos.get('positionAmt', 0)
                if position_amt:
                    try:
                        contracts = float(position_amt)
                    except (ValueError, TypeError):
                        contracts = 0
            
            if abs(contracts) > 0:
                # Kiểm tra nếu là USDT-M (không phải COIN-M)
                symbol = pos.get('symbol', '')
                
                # Chuẩn hóa symbol format
                if ':' in symbol:
                    # Format từ ccxt: "BTC/USDT:USDT" -> "BTC/USDT"
                    symbol = symbol.replace('/USDT:USDT', '/USDT').split(':')[0]
                elif not '/' in symbol and 'USDT' in symbol:
                    # Format "DASHUSDT" -> "DASH/USDT"
                    symbol = symbol.replace('USDT', '/USDT')
                
                # Chỉ lấy USDT-M positions
                if '/USDT' in symbol or symbol.endswith('USDT'):
                    entry_price = float(pos.get('entryPrice', 0) or 0)
                    
                    # Debug: in ra dữ liệu raw nếu cần
                    if debug:
                        print(f"\n[DEBUG] Position data for {symbol}:")
                        print(f"  contracts: {contracts}")
                        print(f"  positionSide: {pos.get('positionSide')}")
                        print(f"  side: {pos.get('side')}")
                        print(f"  info: {pos.get('info', {}).get('positionSide', 'N/A') if pos.get('info') else 'N/A'}")
                        print(f"  info.positionAmt: {pos.get('info', {}).get('positionAmt', 'N/A') if pos.get('info') else 'N/A'}")
                    
                    # Xác định direction (LONG/SHORT)
                    # Binance API có thể trả về positionSide hoặc side
                    # Nếu không có, dựa vào dấu của contracts (positionAmt)
                    direction = None
                    
                    # Kiểm tra positionSide từ API response
                    position_side = pos.get('positionSide', '')
                    if position_side:
                        position_side = str(position_side).upper()
                        if position_side in ['LONG', 'SHORT']:
                            direction = position_side
                    
                    # Kiểm tra side
                    if direction is None and pos.get('side'):
                        side = str(pos.get('side', '')).upper()
                        if side in ['LONG', 'SHORT']:
                            direction = side
                    
                    # Kiểm tra trong info (raw API response từ Binance)
                    if direction is None and pos.get('info'):
                        info = pos.get('info', {})
                        if isinstance(info, dict):
                            # Binance API raw response có thể có positionSide
                            raw_position_side = info.get('positionSide', '')
                            if raw_position_side:
                                raw_position_side = str(raw_position_side).upper()
                                if raw_position_side in ['LONG', 'SHORT']:
                                    direction = raw_position_side
                            
                            # Hoặc kiểm tra positionAmt trong info
                            raw_position_amt = info.get('positionAmt', '')
                            if raw_position_amt:
                                try:
                                    raw_amt = float(raw_position_amt)
                                    if raw_amt != 0:
                                        direction = 'LONG' if raw_amt > 0 else 'SHORT'
                                except (ValueError, TypeError):
                                    pass
                    
                    # Fallback cuối cùng: dựa vào dấu của contracts
                    # contracts > 0 = LONG, contracts < 0 = SHORT
                    if direction is None:
                        direction = 'LONG' if contracts > 0 else 'SHORT'
                    
                    # Lấy notional value (size theo USDT) từ position
                    # Với Binance Futures USDT-M:
                    # - contracts (positionAmt) là số lượng contracts (có thể âm)
                    # - entryPrice là giá vào lệnh
                    # - notional = abs(contracts) * entryPrice (size theo USDT)
                    
                    # Kiểm tra nếu có field 'notional' trực tiếp
                    notional = pos.get('notional', None)
                    if notional is not None and notional != 0:
                        size_usdt = abs(float(notional))
                    else:
                        # Tính từ contracts * entryPrice
                        # Với USDT-M: notional = abs(contracts) * entryPrice
                        size_usdt = abs(contracts * entry_price)
                        
                        # Nếu có vấn đề, thử fetch position detail để lấy chính xác hơn
                        if size_usdt == 0 and entry_price > 0:
                            try:
                                pos_detail = exchange.fetch_position(symbol)
                                notional = pos_detail.get('notional', None)
                                if notional is not None and notional != 0:
                                    size_usdt = abs(float(notional))
                            except:
                                pass
                    
                    open_positions.append({
                        'symbol': symbol if '/' in symbol else symbol.replace('USDT', '/USDT'),
                        'size_usdt': size_usdt,
                        'entry_price': entry_price,
                        'direction': direction,
                        'contracts': abs(contracts),
                    })
        
        return open_positions
        
    except ccxt.AuthenticationError as e:
        raise ValueError(f"Lỗi xác thực API: {e}\nVui lòng kiểm tra lại API Key và Secret")
    except ccxt.NetworkError as e:
        raise ValueError(f"Lỗi kết nối mạng: {e}")
    except Exception as e:
        raise ValueError(f"Lỗi khi lấy positions: {e}")


def format_position_output(positions: List[Dict]) -> str:
    """
    Format danh sách positions theo định dạng yêu cầu
    
    Args:
        positions: List các dictionary chứa thông tin position
    
    Returns:
        String đã được format
    """
    if not positions:
        return f"{Fore.YELLOW}Không có position nào đang mở.{Style.RESET_ALL}"
    
    output_lines = []
    for pos in positions:
        symbol = pos['symbol']
        direction = pos['direction']
        entry_price = pos['entry_price']
        size_usdt = pos['size_usdt']
        
        # Format theo yêu cầu: SYMBOL/USDT  DIRECTION Entry:  ENTRY_PRICE Size:  SIZE USDT
        line = f"{symbol:<15} {direction:<5} Entry: {entry_price:>12.8f} Size: {size_usdt:>12.2f} USDT"
        output_lines.append(line)
    
    return '\n'.join(output_lines)


def main():
    """Hàm main để chạy chương trình"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Lấy thông tin các position đang mở từ Binance Futures USDT-M',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Ví dụ sử dụng:
            # Sử dụng API key từ modules/config_api.py (đặt BINANCE_API_KEY và BINANCE_API_SECRET trong file):
            $ python -m modules.binance_positions
            
            # Hoặc sử dụng từ biến môi trường:
            $ set BINANCE_API_KEY=your_key
            $ set BINANCE_API_SECRET=your_secret
            $ python -m modules.binance_positions
            
            # Hoặc cung cấp trực tiếp qua command line:
            $ python -m modules.binance_positions --api-key YOUR_KEY --api-secret YOUR_SECRET
            
            # Sử dụng testnet:
            $ python -m modules.binance_positions --testnet

            Để lấy API Key:
            1. Đăng nhập vào Binance
            2. Vào API Management
            3. Tạo API Key mới với quyền Futures Trading
            4. Lưu ý: Bật "Enable Futures" trong API settings
            
            Cách cấu hình (theo thứ tự ưu tiên):
            1. Command line arguments (--api-key, --api-secret)
            2. Biến môi trường (BINANCE_API_KEY, BINANCE_API_SECRET)
            3. File config (modules/config_api.py - file này đã được ignore trong .gitignore)
                    """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Binance API Key (hoặc đặt trong modules/config_api.py hoặc biến môi trường BINANCE_API_KEY)'
    )
    
    parser.add_argument(
        '--api-secret',
        type=str,
        default=None,
        help='Binance API Secret (hoặc đặt trong modules/config_api.py hoặc biến môi trường BINANCE_API_SECRET)'
    )
    
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Sử dụng Binance Testnet (mặc định: False)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"{Fore.CYAN}{Style.BRIGHT}Đang kết nối với Binance Futures USDT-M...{Style.RESET_ALL}\n")
        
        # Lấy positions
        positions = get_binance_futures_positions(
            api_key=args.api_key,
            api_secret=args.api_secret,
            testnet=args.testnet
        )
        
        if not positions:
            print(f"{Fore.YELLOW}Không có position nào đang mở.{Style.RESET_ALL}")
            return
        
        # In kết quả
        print(f"{Fore.GREEN}{Style.BRIGHT}=== CÁC POSITION ĐANG MỞ ==={Style.RESET_ALL}\n")
        output = format_position_output(positions)
        print(output)
        print(f"\n{Fore.CYAN}Tổng số position: {len(positions)}{Style.RESET_ALL}")
        
    except ValueError as e:
        print(f"{Fore.RED}{Style.BRIGHT}Lỗi: {e}{Style.RESET_ALL}")
        return 1
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Đã hủy bởi người dùng.{Style.RESET_ALL}")
        return 1
    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}Lỗi không mong đợi: {e}{Style.RESET_ALL}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

