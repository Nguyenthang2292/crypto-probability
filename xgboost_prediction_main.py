import warnings

import numpy as np
from colorama import Fore, Style, init as colorama_init

# Import from xgboost_prediction_ modules (modules specific to xgboost_prediction_main.py)
from modules.config import (
    DEFAULT_SYMBOL,
    DEFAULT_QUOTE,
    DEFAULT_TIMEFRAME,
    DEFAULT_LIMIT,
    DEFAULT_EXCHANGE_STRING,
    DEFAULT_EXCHANGES,
    TARGET_HORIZON,
    TARGET_BASE_THRESHOLD,
    TARGET_LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
)
from modules.utils import color_text, format_price
from modules.xgboost_prediction_utils import get_prediction_window
from modules.xgboost_prediction_cli import parse_args, resolve_input
from modules.xgboost_prediction_data_fetcher import fetch_data_from_ccxt
from modules.utils import normalize_symbol
from modules.xgboost_prediction_indicators import calculate_indicators
from modules.xgboost_prediction_labeling import apply_directional_labels
from modules.xgboost_prediction_model import train_and_predict, predict_next_move

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

def main():
    args = parse_args()
    allow_prompt = not args.no_prompt

    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    timeframe = resolve_input(
        args.timeframe, DEFAULT_TIMEFRAME, "Enter timeframe", str, allow_prompt
    ).lower()
    limit = args.limit if args.limit is not None else DEFAULT_LIMIT
    exchanges_input = args.exchanges if args.exchanges else DEFAULT_EXCHANGE_STRING
    exchanges = [
        ex.strip() for ex in exchanges_input.split(",") if ex.strip()
    ] or DEFAULT_EXCHANGES

    def run_once(raw_symbol):
        symbol = normalize_symbol(raw_symbol, quote)
        df, exchange_id = fetch_data_from_ccxt(
            symbol, timeframe, limit=limit, exchanges=exchanges
        )
        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

        if df is not None:
            # Calculate indicators without labels first (to preserve latest_data)
            df = calculate_indicators(df, apply_labels=False)
            
            # Save latest data before applying labels and dropping NaN
            latest_data = df.iloc[-1:].copy()
            # Fill any remaining NaN in latest_data with forward fill then backward fill
            latest_data = latest_data.ffill().bfill()

            # Apply directional labels and drop NaN for training data
            df = apply_directional_labels(df)
            latest_threshold = df["DynamicThreshold"].iloc[-1] if len(df) > 0 else TARGET_BASE_THRESHOLD
            df.dropna(inplace=True)
            latest_data["DynamicThreshold"] = latest_threshold

            print(color_text(f"Training on {len(df)} samples...", Fore.CYAN))
            model = train_and_predict(df)

            proba = predict_next_move(model, latest_data)
            proba_percent = {
                label: proba[LABEL_TO_ID[label]] * 100 for label in TARGET_LABELS
            }
            best_idx = int(np.argmax(proba))
            direction = ID_TO_LABEL[best_idx]
            probability = proba_percent[direction]

            current_price = latest_data["close"].values[0]
            atr = latest_data["ATR_14"].values[0]
            prediction_window = get_prediction_window(timeframe)
            threshold_value = latest_data["DynamicThreshold"].iloc[0]
            prediction_context = f"{prediction_window} | {TARGET_HORIZON} candles >={threshold_value*100:.2f}% move"

            print("\n" + color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
            print(
                color_text(
                    f"ANALYSIS FOR {symbol} | TF {timeframe} | {exchange_label}",
                    Fore.CYAN,
                    Style.BRIGHT,
                )
            )
            print(
                color_text(f"Current Price: {format_price(current_price)}", Fore.WHITE)
            )
            print(
                color_text(f"Market Volatility (ATR): {format_price(atr)}", Fore.WHITE)
            )
            print(color_text("-" * 40, Fore.BLUE))

            if direction == "UP":
                direction_color = Fore.GREEN
                atr_sign = 1
            elif direction == "DOWN":
                direction_color = Fore.RED
                atr_sign = -1
            else:
                direction_color = Fore.YELLOW
                atr_sign = 0

            print(
                color_text(
                    f"PREDICTION ({prediction_context}): {direction}",
                    direction_color,
                    Style.BRIGHT,
                )
            )
            print(color_text(f"Confidence: {probability:.2f}%", direction_color))

            prob_summary = " | ".join(
                f"{label}: {value:.2f}%" for label, value in proba_percent.items()
            )
            print(color_text(f"Probabilities -> {prob_summary}", Fore.WHITE))

            if direction == "NEUTRAL":
                print(
                    color_text(
                        "Market expected to stay within +/-{:.2f}% over the next {} candles.".format(
                            threshold_value * 100, TARGET_HORIZON
                        ),
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "Estimated Targets via ATR multiples:",
                        Fore.MAGENTA,
                        Style.BRIGHT,
                    )
                )
                for multiple in (1, 2, 3):
                    target_price = current_price + atr_sign * multiple * atr
                    move_abs = abs(target_price - current_price)
                    move_pct = (
                        (move_abs / current_price) * 100 if current_price else None
                    )
                    move_pct_text = (
                        f"{move_pct:.2f}%" if move_pct is not None else "N/A"
                    )
                    print(
                        color_text(
                            f"  ATR x{multiple}: {format_price(target_price)} | Delta {format_price(move_abs)} ({move_pct_text})",
                            Fore.MAGENTA,
                        )
                    )
            print(color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
        else:
            print(
                color_text(
                    "Unable to proceed without market data. Please try again later.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )

    try:
        while True:
            raw_symbol = resolve_input(
                args.symbol, DEFAULT_SYMBOL, "Enter symbol pair", str, allow_prompt
            )
            run_once(raw_symbol)
            args.symbol = None  # force prompt next iteration
            if not allow_prompt:
                break
            print(
                color_text(
                    "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                    Fore.YELLOW,
                )
            )
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))

if __name__ == "__main__":
    main()

