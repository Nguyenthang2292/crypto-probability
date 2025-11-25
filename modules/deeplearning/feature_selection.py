"""
Feature Selection & Engineering Module for Deep Learning Pipeline.

This module provides comprehensive feature selection including:
- Mutual Information based feature selection
- Boruta-like feature selection (using Random Forest importance)
- Collinearity removal to improve model stability
- Feature filtering to avoid "Garbage In, Garbage Out"
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    SelectKBest,
    f_regression,
    f_classif,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from colorama import Fore, Style

from modules.config import (
    DEEP_FEATURE_SELECTION_METHOD,
    DEEP_FEATURE_SELECTION_TOP_K,
    DEEP_FEATURE_COLLINEARITY_THRESHOLD,
    DEEP_FEATURE_SELECTION_DIR,
)
from modules.common.utils import color_text


class FeatureSelector:
    """
    Comprehensive feature selection and engineering for deep learning models.
    
    Methods:
    1. Mutual Information: Selects features with highest mutual information with target
    2. Boruta-like: Uses Random Forest importance to select relevant features
    3. Collinearity Removal: Removes highly correlated features
    """

    def __init__(
        self,
        method: str = DEEP_FEATURE_SELECTION_METHOD,
        top_k: int = DEEP_FEATURE_SELECTION_TOP_K,
        collinearity_threshold: float = DEEP_FEATURE_COLLINEARITY_THRESHOLD,
        selection_dir: str = DEEP_FEATURE_SELECTION_DIR,
    ):
        """
        Args:
            method: Feature selection method ('mutual_info', 'boruta', 'f_test', or 'combined')
            top_k: Number of top features to select (20-30 recommended)
            collinearity_threshold: Correlation threshold for removing collinear features (0.8-0.95)
            selection_dir: Directory to save/load feature selection results
        """
        self.method = method
        self.top_k = top_k
        self.collinearity_threshold = collinearity_threshold
        self.selection_dir = Path(selection_dir)
        self.selection_dir.mkdir(parents=True, exist_ok=True)

        # Store selected features
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.selection_metadata: Dict = {}

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "regression",
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Select top features using specified method.

        Args:
            X: Feature DataFrame
            y: Target Series (can be continuous for regression or discrete for classification)
            task_type: 'regression' or 'classification'
            symbol: Optional symbol name for saving per-symbol selections

        Returns:
            DataFrame with selected features only
        """
        # Step 1: Remove non-numeric and invalid columns
        X_clean = self._filter_invalid_features(X, y)

        if X_clean.empty:
            raise ValueError("No valid features after filtering")

        # Step 2: Remove highly collinear features
        X_clean = self._remove_collinear_features(X_clean)

        # Step 3: Apply feature selection method
        if self.method == "mutual_info":
            selected_features = self._select_by_mutual_info(
                X_clean, y, task_type
            )
        elif self.method == "boruta":
            selected_features = self._select_by_boruta_like(
                X_clean, y, task_type
            )
        elif self.method == "f_test":
            selected_features = self._select_by_f_test(X_clean, y, task_type)
        elif self.method == "combined":
            selected_features = self._select_combined(X_clean, y, task_type)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'mutual_info', 'boruta', 'f_test', or 'combined'"
            )

        # Step 4: Store results
        self.selected_features = selected_features
        self._save_selection(symbol)

        print(
            color_text(
                f"Selected {len(selected_features)} features using {self.method}",
                Fore.GREEN,
            )
        )

        return X_clean[selected_features]

    def _filter_invalid_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """
        Filter out invalid features:
        - Non-numeric columns
        - Columns with too many NaN values
        - Constant columns (zero variance)
        - Target leakage columns
        """
        X_clean = X.copy()

        # Remove non-numeric columns
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        X_clean = X_clean[numeric_cols]

        # Remove columns with >50% NaN values
        nan_threshold = 0.5
        valid_cols = X_clean.columns[
            X_clean.isna().sum() / len(X_clean) < nan_threshold
        ].tolist()
        X_clean = X_clean[valid_cols]

        # Remove constant columns (zero or near-zero variance)
        constant_cols = []
        for col in X_clean.columns:
            if X_clean[col].nunique() <= 1:
                constant_cols.append(col)
            elif X_clean[col].std() < 1e-8:
                constant_cols.append(col)

        if constant_cols:
            print(
                color_text(
                    f"Removing {len(constant_cols)} constant features",
                    Fore.YELLOW,
                )
            )
            X_clean = X_clean.drop(columns=constant_cols)

        # Remove target leakage columns (columns that contain future information)
        leakage_keywords = [
            "future_",
            "target",
            "label",
            "triple_barrier",
        ]
        leakage_cols = [
            col
            for col in X_clean.columns
            if any(keyword in col.lower() for keyword in leakage_keywords)
        ]

        if leakage_cols:
            print(
                color_text(
                    f"Removing {len(leakage_cols)} potential target leakage features",
                    Fore.YELLOW,
                )
            )
            X_clean = X_clean.drop(columns=leakage_cols)

        # Remove timestamp and symbol columns if present
        exclude_cols = ["timestamp", "symbol", "time_idx"]
        exclude_cols = [col for col in exclude_cols if col in X_clean.columns]
        if exclude_cols:
            X_clean = X_clean.drop(columns=exclude_cols)

        return X_clean

    def _remove_collinear_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly collinear features using correlation matrix.
        Keeps the feature with highest variance among correlated pairs.
        """
        if len(X.columns) <= 1:
            return X

        # Compute correlation matrix
        corr_matrix = X.corr().abs()

        # Find pairs with correlation above threshold
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = []
        for col in upper_triangle.columns:
            if col in to_drop:
                continue
            highly_correlated = upper_triangle.index[
                upper_triangle[col] > self.collinearity_threshold
            ].tolist()

            if highly_correlated:
                # Keep the feature with highest variance
                candidates = [col] + highly_correlated
                variances = X[candidates].var()
                keep_feature = variances.idxmax()
                drop_features = [c for c in candidates if c != keep_feature]
                to_drop.extend(drop_features)

        if to_drop:
            print(
                color_text(
                    f"Removing {len(to_drop)} collinear features (threshold={self.collinearity_threshold})",
                    Fore.YELLOW,
                )
            )
            X_clean = X.drop(columns=to_drop)
        else:
            X_clean = X

        return X_clean

    def _select_by_mutual_info(
        self, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> List[str]:
        """Select features using Mutual Information."""
        # Handle NaN values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median() if task_type == "regression" else y.mode()[0])

        if task_type == "regression":
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
        else:
            # For classification, ensure y is integer-encoded
            if not pd.api.types.is_integer_dtype(y_clean):
                le = LabelEncoder()
                y_clean = le.fit_transform(y_clean)
            mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)

        # Get top K features
        feature_scores = dict(zip(X.columns, mi_scores))
        self.feature_scores = feature_scores

        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = [feat for feat, score in sorted_features[: self.top_k]]

        return selected

    def _select_by_boruta_like(
        self, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> List[str]:
        """
        Boruta-like feature selection using Random Forest importance.
        This is a simplified version that uses RF importance scores.
        """
        # Handle NaN values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median() if task_type == "regression" else y.mode()[0])

        if task_type == "regression":
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
            )
        else:
            if not pd.api.types.is_integer_dtype(y_clean):
                le = LabelEncoder()
                y_clean = le.fit_transform(y_clean)
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
            )

        model.fit(X_clean, y_clean)

        # Get feature importances
        importances = model.feature_importances_
        feature_scores = dict(zip(X.columns, importances))
        self.feature_scores = feature_scores

        # Select top K features
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = [feat for feat, score in sorted_features[: self.top_k]]

        return selected

    def _select_by_f_test(
        self, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> List[str]:
        """Select features using F-test (ANOVA F-statistic)."""
        # Handle NaN values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median() if task_type == "regression" else y.mode()[0])

        if task_type == "regression":
            selector = SelectKBest(score_func=f_regression, k=self.top_k)
        else:
            if not pd.api.types.is_integer_dtype(y_clean):
                le = LabelEncoder()
                y_clean = le.fit_transform(y_clean)
            selector = SelectKBest(score_func=f_classif, k=self.top_k)

        selector.fit(X_clean, y_clean)

        # Get selected features
        selected = X.columns[selector.get_support()].tolist()

        # Store scores
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.feature_scores = feature_scores

        return selected

    def _select_combined(
        self, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> List[str]:
        """
        Combined approach: Use both Mutual Information and Boruta-like,
        then take intersection or union of top features.
        """
        # Get features from both methods
        mi_features = self._select_by_mutual_info(X, y, task_type)
        mi_scores = self.feature_scores.copy()

        boruta_features = self._select_by_boruta_like(X, y, task_type)
        boruta_scores = self.feature_scores.copy()

        # Combine scores (normalized average)
        all_features = set(mi_features + boruta_features)
        combined_scores = {}

        for feat in all_features:
            mi_score = mi_scores.get(feat, 0)
            boruta_score = boruta_scores.get(feat, 0)

            # Normalize scores to [0, 1] range
            mi_norm = (
                (mi_score - min(mi_scores.values()))
                / (max(mi_scores.values()) - min(mi_scores.values()) + 1e-8)
                if max(mi_scores.values()) > min(mi_scores.values())
                else 0
            )
            boruta_norm = (
                (boruta_score - min(boruta_scores.values()))
                / (max(boruta_scores.values()) - min(boruta_scores.values()) + 1e-8)
                if max(boruta_scores.values()) > min(boruta_scores.values())
                else 0
            )

            # Average normalized scores
            combined_scores[feat] = (mi_norm + boruta_norm) / 2

        self.feature_scores = combined_scores

        # Select top K based on combined scores
        sorted_features = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = [feat for feat, score in sorted_features[: self.top_k]]

        return selected

    def _save_selection(self, symbol: Optional[str] = None) -> None:
        """Save feature selection results to disk."""
        metadata = {
            "method": self.method,
            "top_k": self.top_k,
            "collinearity_threshold": self.collinearity_threshold,
            "selected_features": self.selected_features,
            "feature_scores": {
                k: float(v) for k, v in self.feature_scores.items()
            },
        }

        self.selection_metadata = metadata

        # Save to file
        filename = f"feature_selection_{symbol or 'default'}.json"
        filepath = self.selection_dir / filename

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_selection(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Load feature selection results from disk."""
        filename = f"feature_selection_{symbol or 'default'}.json"
        filepath = self.selection_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            metadata = json.load(f)

        self.selected_features = metadata["selected_features"]
        self.feature_scores = metadata["feature_scores"]
        self.selection_metadata = metadata

        return metadata

    def apply_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply previously selected features to a new DataFrame.
        Must call select_features() or load_selection() first.
        """
        if not self.selected_features:
            raise ValueError(
                "No features selected. Call select_features() or load_selection() first."
            )

        # Get available features (some might be missing in new data)
        available_features = [
            feat for feat in self.selected_features if feat in X.columns
        ]

        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            print(
                color_text(
                    f"Warning: {len(missing)} selected features missing in data",
                    Fore.YELLOW,
                )
            )

        return X[available_features]

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Get a DataFrame with feature importance scores.
        """
        if not self.feature_scores:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "feature": list(self.feature_scores.keys()),
                "score": list(self.feature_scores.values()),
                "selected": [
                    feat in self.selected_features
                    for feat in self.feature_scores.keys()
                ],
            }
        )

        df = df.sort_values("score", ascending=False)
        return df

