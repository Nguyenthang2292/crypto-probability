"""
Direction metrics calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

try:
    from modules.config import (
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES,
    )
except ImportError:
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES = 20


def calculate_direction_metrics(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    classification_zscore: float = PAIRS_TRADING_CLASSIFICATION_ZSCORE,
) -> Dict[str, Optional[float]]:
    """
    Calculate classification metrics for spread direction prediction using z-score.
    
    Evaluates how well z-score predicts future spread direction for BOTH Long and Short signals.
    Metrics are calculated ONLY on active signals (when z-score exceeds threshold), 
    ignoring neutral periods.
    
    Logic:
    - **Long Signal**: zscore < -threshold → Predict UP (1). Correct if spread increases (actual = 1).
    - **Short Signal**: zscore > threshold → Predict DOWN (-1). Correct if spread decreases (actual = -1).
    - **Neutral**: -threshold <= zscore <= threshold → No prediction (0). Ignored in metrics.
    
    Metrics Calculation:
    - **Precision**: Calculated separately for Long and Short, then macro-averaged.
      - Long Precision = Correct Long predictions / All Long predictions
      - Short Precision = Correct Short predictions / All Short predictions
    - **Recall**: Calculated separately for Long and Short, then macro-averaged.
      - Long Recall = Correct Long predictions / All actual Long movements (in active set)
      - Short Recall = Correct Short predictions / All actual Short movements (in active set)
    - **F1**: Harmonic mean of macro-averaged precision and recall.
    - **Accuracy**: Overall accuracy = Correct predictions / Total active signals.
    
    Args:
        spread: Spread series (pd.Series, price1 - hedge_ratio * price2)
        zscore_lookback: Rolling window size for z-score calculation (must be > 0). Default: 60
        classification_zscore: Z-score threshold for prediction (must be > 0). Default: 0.5
        
    Returns:
        Dict with metrics (all in [0, 1] or None):
        - classification_accuracy: Overall accuracy of all active signals.
        - classification_precision: Macro-averaged precision (average of Long and Short precision).
        - classification_recall: Macro-averaged recall (average of Long and Short recall).
        - classification_f1: Harmonic mean of macro-averaged precision and recall.
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, ...])
        >>> metrics = calculate_direction_metrics(spread)
        >>> # Returns dict with accuracy, precision, recall, f1 scores
    """
    result = {
        "classification_f1": None,
        "classification_precision": None,
        "classification_recall": None,
        "classification_accuracy": None,
    }

    if spread is None:
        return result
    
    if not isinstance(spread, pd.Series):
        return result
    
    if len(spread) < zscore_lookback:
        return result
    
    # Validate zscore_lookback
    if zscore_lookback <= 0:
        return result

    # Handle NaN values: drop NaN to ensure clean calculations
    # This ensures rolling window calculations are based on valid data only
    spread_clean = spread.dropna()
    
    # Check if we have enough valid data points after removing NaN
    if len(spread_clean) < zscore_lookback:
        return result
    
    # Validate classification_zscore
    if classification_zscore <= 0:
        return result
    
    # Calculate rolling z-score on clean data
    rolling_mean = spread_clean.rolling(zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread_clean.rolling(zscore_lookback, min_periods=zscore_lookback).std().replace(0, np.nan)
    zscore = ((spread_clean - rolling_mean) / rolling_std).dropna()
    
    # Validate zscore has enough valid values
    if zscore.empty or len(zscore) < 2:
        return result
    
    # Actual direction: 1 if spread increases, -1 if decreases, 0 if unchanged
    future_return = spread_clean.shift(-1) - spread_clean
    actual_direction = np.sign(future_return).dropna()
    
    # Validate actual_direction has enough valid values
    if actual_direction.empty or len(actual_direction) < 2:
        return result
    
    # Edge case: If spread never changes (all future_return = 0), cannot calculate meaningful metrics
    if (actual_direction == 0).all():
        return result

    # Align indices
    common_idx = zscore.index.intersection(actual_direction.index)
    
    # Validate we have enough common indices
    if len(common_idx) < 2:
        return result
    
    zscore = zscore.loc[common_idx]
    actual_direction = actual_direction.loc[common_idx]

    # Prediction logic (3-class system):
    # 1 (Long) if zscore < -threshold
    # -1 (Short) if zscore > threshold
    # 0 (Neutral) otherwise
    threshold = classification_zscore
    predicted_signal = np.select(
        [zscore < -threshold, zscore > threshold], 
        [1.0, -1.0], 
        default=0.0
    )
    
    # Filter only active signals (ignore Neutral 0s)
    active_mask = predicted_signal != 0
    
    # Require minimum samples of ACTIVE SIGNALS for reliable metrics
    # If we have very few trades, metrics are noisy
    if active_mask.sum() < PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES:
        return result

    # Get active predictions and actual outcomes
    active_pred = predicted_signal[active_mask]
    active_actual = actual_direction[active_mask]
    
    # Calculate Accuracy: Fraction of times Signal matches Direction
    # Note: actual_direction can be 0 (unchanged), which counts as wrong for both Long/Short
    correct_predictions = (active_pred == active_actual)
    accuracy = correct_predictions.mean()
    
    # Validate accuracy is finite and in [0, 1]
    if np.isnan(accuracy) or np.isinf(accuracy) or accuracy < 0 or accuracy > 1:
        accuracy = None
    
    try:
        # Calculate metrics separately for Long and Short signals
        # Long signals: predicted = 1, correct when actual = 1
        long_mask = active_pred == 1.0
        long_predicted = long_mask.sum()
        
        # Short signals: predicted = -1, correct when actual = -1
        short_mask = active_pred == -1.0
        short_predicted = short_mask.sum()
        
        # Calculate Long metrics
        long_precision = None
        long_recall = None
        long_f1 = None
        
        if long_predicted > 0:
            # Long Precision: TP / (TP + FP) = correct Long predictions / all Long predictions
            long_correct = ((active_pred == 1.0) & (active_actual == 1.0)).sum()
            long_precision = long_correct / long_predicted if long_predicted > 0 else 0.0
            
            # Long Recall: TP / (TP + FN) = correct Long predictions / all actual Long movements
            # Note: We only consider active signals, so actual Long = actual_direction == 1 in active set
            long_actual = (active_actual == 1.0).sum()
            long_recall = long_correct / long_actual if long_actual > 0 else 0.0
            
            # Long F1: harmonic mean of precision and recall
            if long_precision > 0 and long_recall > 0:
                long_f1 = 2 * (long_precision * long_recall) / (long_precision + long_recall)
            else:
                long_f1 = 0.0
        
        # Calculate Short metrics
        short_precision = None
        short_recall = None
        short_f1 = None
        
        if short_predicted > 0:
            # Short Precision: TP / (TP + FP) = correct Short predictions / all Short predictions
            short_correct = ((active_pred == -1.0) & (active_actual == -1.0)).sum()
            short_precision = short_correct / short_predicted if short_predicted > 0 else 0.0
            
            # Short Recall: TP / (TP + FN) = correct Short predictions / all actual Short movements
            short_actual = (active_actual == -1.0).sum()
            short_recall = short_correct / short_actual if short_actual > 0 else 0.0
            
            # Short F1: harmonic mean of precision and recall
            if short_precision > 0 and short_recall > 0:
                short_f1 = 2 * (short_precision * short_recall) / (short_precision + short_recall)
            else:
                short_f1 = 0.0
        
        # Calculate macro-averaged metrics (average of Long and Short)
        # If one class is missing, use the other class's metrics
        precision_values = [v for v in [long_precision, short_precision] if v is not None]
        recall_values = [v for v in [long_recall, short_recall] if v is not None]
        
        macro_precision = None
        macro_recall = None
        
        if precision_values:
            macro_precision_val = np.mean(precision_values)
            # Validate macro_precision is finite and in [0, 1]
            if not np.isnan(macro_precision_val) and not np.isinf(macro_precision_val) and 0 <= macro_precision_val <= 1:
                macro_precision = float(macro_precision_val)
        
        if recall_values:
            macro_recall_val = np.mean(recall_values)
            # Validate macro_recall is finite and in [0, 1]
            if not np.isnan(macro_recall_val) and not np.isinf(macro_recall_val) and 0 <= macro_recall_val <= 1:
                macro_recall = float(macro_recall_val)
        
        # Calculate macro F1 from macro precision and recall (more consistent)
        if macro_precision is not None and macro_recall is not None:
            if macro_precision > 0 and macro_recall > 0:
                macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
            else:
                macro_f1 = 0.0
        else:
            macro_f1 = None
        
        # Set results (use macro-averaged metrics)
        # Validate all metrics are in [0, 1] before setting
        if accuracy is not None and 0 <= accuracy <= 1:
            result["classification_accuracy"] = float(accuracy)
        else:
            result["classification_accuracy"] = None
        result["classification_precision"] = macro_precision
        result["classification_recall"] = macro_recall
        result["classification_f1"] = macro_f1
        
        # Final validation: ensure F1 is in [0, 1] if not None
        if result["classification_f1"] is not None:
            if result["classification_f1"] < 0 or result["classification_f1"] > 1:
                result["classification_f1"] = None
        
        return result
    except (ValueError, ZeroDivisionError, AttributeError, TypeError, IndexError, KeyError):
        # ValueError: Invalid values in calculations (e.g., NaN in mean)
        # ZeroDivisionError: Division by zero in precision/recall calculations
        # AttributeError: Missing attributes on pandas Series/DataFrame
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # IndexError: Index access errors (e.g., empty Series)
        # KeyError: Dictionary key access errors (shouldn't happen, but safety)
        return {
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
        }

