"""
Model training and prediction functions for xgboost_prediction_main.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from .utils import color_text
from colorama import Fore, Style
from .config import TARGET_LABELS, TARGET_HORIZON, MODEL_FEATURES, ID_TO_LABEL, XGBOOST_PARAMS
from .xgboost_prediction_display import print_classification_report


def train_and_predict(df):
    """
    Trains XGBoost model and predicts the next movement.
    """
    X = df[MODEL_FEATURES]
    y = df["Target"].astype(int)

    def build_model():
        # Use parameters from config, adding num_class dynamically
        params = XGBOOST_PARAMS.copy()
        params["num_class"] = len(TARGET_LABELS)
        return xgb.XGBClassifier(**params)

    # Train/Test split (80/20) for evaluation metrics
    # IMPORTANT: Create a gap of TARGET_HORIZON between train and test to prevent data leakage
    # The last TARGET_HORIZON rows of train set use future prices from test set to create labels
    split = int(len(df) * 0.8)
    train_end = split - TARGET_HORIZON
    test_start = split
    
    # Ensure we have enough data after creating the gap
    if train_end < len(df) * 0.5:
        train_end = int(len(df) * 0.5)
        test_start = train_end + TARGET_HORIZON
        if test_start >= len(df):
            # Not enough data for proper train/test split with gap
            print(
                color_text(
                    f"WARNING: Insufficient data for train/test split with gap. "
                    f"Need at least {len(df) + TARGET_HORIZON} rows. Using all data for training.",
                    Fore.YELLOW,
                )
            )
            train_end = len(df)
            test_start = len(df)
    
    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]
    
    gap_size = test_start - train_end
    if gap_size > 0:
        print(
            color_text(
                f"Train/Test split: {len(X_train)} train, {gap_size} gap (to prevent leakage), {len(X_test)} test",
                Fore.CYAN,
            )
        )

    model = build_model()
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        print(color_text(f"\nHoldout Accuracy: {score:.4f}", Fore.YELLOW, Style.BRIGHT))
        print_classification_report(y_test, y_pred, "Holdout Test Set Evaluation")
    else:
        print(
            color_text(
                "Skipping holdout evaluation (insufficient test data after gap).",
                Fore.YELLOW,
            )
        )

    # Time-series cross validation with gap to prevent data leakage
    max_splits = min(5, len(df) - 1)
    if max_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=max_splits)
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            # Apply gap to prevent data leakage: remove last TARGET_HORIZON indices from train
            train_idx_array = np.array(train_idx)
            if len(train_idx_array) > TARGET_HORIZON:
                # Remove the last TARGET_HORIZON indices from train to create gap
                train_idx_filtered = train_idx_array[:-TARGET_HORIZON]
            else:
                # Not enough data for gap, skip this fold
                print(
                    color_text(
                        f"CV Fold {fold}: Skipped (insufficient train data for gap)",
                        Fore.YELLOW,
                    )
                )
                continue
            
            # Ensure test set doesn't overlap with gap
            # Gap is sufficient when: test_start > train_end + TARGET_HORIZON
            test_idx_array = np.array(test_idx)
            if len(train_idx_filtered) > 0 and len(test_idx_array) > 0:
                min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
                if test_idx_array[0] < min_test_start:
                    # Adjust test start to create proper gap
                    test_idx_array = test_idx_array[test_idx_array >= min_test_start]
                    if len(test_idx_array) == 0:
                        print(
                            color_text(
                                f"CV Fold {fold}: Skipped (no valid test data after gap)",
                                Fore.YELLOW,
                            )
                        )
                        continue
            
            # Check if filtered training data contains all required classes
            y_train_fold = y.iloc[train_idx_filtered]
            unique_classes = sorted(y_train_fold.unique())
            
            # XGBoost requires at least 2 classes, but we need all 3 for proper multi-class
            # If we don't have all classes, skip this fold
            if len(unique_classes) < 2:
                print(
                    color_text(
                        f"CV Fold {fold}: Skipped (insufficient class diversity: {unique_classes})",
                        Fore.YELLOW,
                    )
                )
                continue
            
            # If we have all 3 classes, proceed normally
            # If we only have 2 classes, we can still train but need to handle it
            # For now, we'll skip folds that don't have all 3 classes to maintain consistency
            if len(unique_classes) < len(TARGET_LABELS):
                print(
                    color_text(
                        f"CV Fold {fold}: Skipped (missing classes: expected {TARGET_LABELS}, got {[ID_TO_LABEL[c] for c in unique_classes]})",
                        Fore.YELLOW,
                    )
                )
                continue
            
            cv_model = build_model()
            cv_model.fit(X.iloc[train_idx_filtered], y.iloc[train_idx_filtered])
            if len(test_idx_array) > 0:
                y_test_fold = y.iloc[test_idx_array]
                preds = cv_model.predict(X.iloc[test_idx_array])
                acc = accuracy_score(y_test_fold, preds)
                cv_scores.append(acc)
                
                # Collect predictions for aggregated report
                all_y_true.extend(y_test_fold.tolist())
                all_y_pred.extend(preds.tolist())
                
                print(
                    color_text(
                        f"CV Fold {fold} Accuracy: {acc:.4f} (train: {len(train_idx_filtered)}, gap: {TARGET_HORIZON}, test: {len(test_idx_array)})",
                        Fore.BLUE,
                    )
                )
        
        if len(cv_scores) > 0:
            mean_cv = sum(cv_scores) / len(cv_scores)
            print(
                color_text(
                    f"\nCV Mean Accuracy ({len(cv_scores)} folds): {mean_cv:.4f}",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
            
            # Print aggregated classification report across all CV folds
            if len(all_y_true) > 0 and len(all_y_pred) > 0:
                print_classification_report(
                    np.array(all_y_true),
                    np.array(all_y_pred),
                    "Cross-Validation Aggregated Report (All Folds)",
                )
        else:
            print(
                color_text(
                    "CV: No valid folds after applying gap. Consider increasing data limit.",
                    Fore.YELLOW,
                )
            )
    else:
        print(
            color_text(
                "Not enough data for cross-validation (requires >=3 samples).",
                Fore.YELLOW,
            )
        )

    model.fit(X, y)
    return model


def predict_next_move(model, last_row):
    """
    Predicts the probability for the next candle.
    """
    X_new = last_row[MODEL_FEATURES].values.reshape(1, -1)
    
    # Predict probability
    proba = model.predict_proba(X_new)[0]
    
    return proba

