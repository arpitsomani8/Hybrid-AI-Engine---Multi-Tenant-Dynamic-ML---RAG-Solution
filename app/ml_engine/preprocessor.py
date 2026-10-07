import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("HybridAIEngine.Preprocessor")

class TabularPreprocessor:
    """
    Automates tabular preprocessing:
    - Identifies numeric and categorical columns
    - Imputes missing values (median for numeric, mode for categorical)
    - Formats categorical columns for native LightGBM handling
    - Computes baseline statistical profiles for drift monitoring
    """
    def __init__(self):
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.impute_values: Dict[str, Any] = {}
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        clean_df = df.copy()

        if target_col and target_col in clean_df.columns:
            y = clean_df[target_col]
            X = clean_df.drop(columns=[target_col])
        else:
            y = None
            X = clean_df

        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Learn imputation and baseline statistics
        for col in self.numerical_cols:
            median_val = float(X[col].median()) if not X[col].dropna().empty else 0.0
            self.impute_values[col] = median_val
            X[col] = X[col].fillna(median_val)

            mean_val = float(X[col].mean())
            std_val = float(X[col].std()) if len(X[col]) > 1 else 1.0
            self.baseline_stats[col] = {
                "mean": round(mean_val, 4),
                "std": round(std_val if std_val > 0 else 1.0, 4),
                "min": round(float(X[col].min()), 4),
                "max": round(float(X[col].max()), 4)
            }

        for col in self.categorical_cols:
            mode_val = str(X[col].mode().iloc[0]) if not X[col].dropna().empty else "missing"
            self.impute_values[col] = mode_val
            X[col] = X[col].fillna(mode_val).astype("category")

        logger.info(f"Fitted preprocessor: {len(self.numerical_cols)} numeric cols, {len(self.categorical_cols)} categorical cols")
        return X, y

    def transform_single(self, row_dict: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
        """
        Transforms a single inference payload dictionary into an aligned DataFrame for LightGBM.
        """
        row_df = pd.DataFrame([row_dict])
        
        # Ensure all expected columns are present
        for col in expected_features:
            if col not in row_df.columns:
                row_df[col] = self.impute_values.get(col, 0.0)

        # Select expected columns
        row_df = row_df[expected_features].copy()

        # Apply learned types and imputations
        for col in expected_features:
            if col in self.categorical_cols:
                if col in self.impute_values:
                    row_df[col] = row_df[col].fillna(self.impute_values[col])
                row_df[col] = row_df[col].astype("category")
            elif col in self.numerical_cols:
                if col in self.impute_values:
                    row_df[col] = row_df[col].fillna(self.impute_values[col])
                row_df[col] = pd.to_numeric(row_df[col], errors="coerce").fillna(self.impute_values.get(col, 0.0))

        return row_df

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "impute_values": self.impute_values,
            "baseline_stats": self.baseline_stats
        }

    def from_dict(self, data: Dict[str, Any]):
        self.numerical_cols = data.get("numerical_cols", [])
        self.categorical_cols = data.get("categorical_cols", [])
        self.impute_values = data.get("impute_values", {})
        self.baseline_stats = data.get("baseline_stats", {})
