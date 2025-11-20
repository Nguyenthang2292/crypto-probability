"""
Position loader for loading positions from Binance.
"""
from typing import List
from colorama import Fore, Style

try:
    from modules.Position import Position
    from modules.utils import color_text
    from modules.binance_positions import get_binance_futures_positions
except ImportError:
    Position = None
    color_text = None
    get_binance_futures_positions = None


class PositionLoader:
    """Loads positions from Binance Futures."""
    
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
    
    def load_from_binance(self, api_key=None, api_secret=None, testnet=None, debug=False):
        """Load positions directly from Binance Futures USDT-M."""
        if get_binance_futures_positions is None:
            raise ImportError("Cannot import get_binance_futures_positions from modules.binance_positions")
        
        if api_key is not None:
            self.api_key = api_key
        if api_secret is not None:
            self.api_secret = api_secret
        if testnet is not None:
            self.testnet = testnet
        
        print(color_text("Loading positions from Binance Futures USDT-M...", Fore.CYAN, Style.BRIGHT))
        
        try:
            binance_positions = get_binance_futures_positions(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                debug=debug
            )
            
            if not binance_positions:
                print(color_text("No open positions found on Binance.", Fore.YELLOW))
                return []
            
            positions = []
            for pos in binance_positions:
                positions.append(Position(
                    symbol=pos['symbol'].upper(),
                    direction=pos['direction'].upper(),
                    entry_price=pos['entry_price'],
                    size_usdt=pos['size_usdt']
                ))
            
            print(color_text(f"✓ Loaded {len(binance_positions)} position(s) from Binance", Fore.GREEN))
            
            print("\n" + color_text("Loaded Positions:", Fore.CYAN))
            for pos in binance_positions:
                print(f"  {pos['symbol']:<15} {pos['direction']:<5} Entry: {pos['entry_price']:>12.8f} Size: {pos['size_usdt']:>12.2f} USDT")
            
            return positions
            
        except Exception as e:
            raise ValueError(f"Error loading positions from Binance: {e}")

