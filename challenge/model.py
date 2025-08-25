# challenge/model.py
import pandas as pd
from typing import Tuple, Union, List, Optional
from datetime import datetime
import xgboost as xgb


class DelayModel:
    """Delay prediction model using XGBoost."""

    _TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(self):
        """Initialize an empty model holder."""
        self._model: Optional[xgb.XGBClassifier] = None

    @staticmethod
    def _is_high_season(ts: pd.Timestamp) -> int:
        """Return 1 if ts is in high season ranges, else 0."""
        if pd.isna(ts):
            return 0
        y = ts.year
        r1_min = pd.Timestamp(year=y, month=12, day=15)
        r1_max = pd.Timestamp(year=y, month=12, day=31, hour=23, minute=59, second=59)
        r2_min = pd.Timestamp(year=y, month=1, day=1)
        r2_max = pd.Timestamp(year=y, month=3, day=3, hour=23, minute=59, second=59)
        r3_min = pd.Timestamp(year=y, month=7, day=15)
        r3_max = pd.Timestamp(year=y, month=7, day=31, hour=23, minute=59, second=59)
        r4_min = pd.Timestamp(year=y, month=9, day=11)
        r4_max = pd.Timestamp(year=y, month=9, day=30, hour=23, minute=59, second=59)
        return int(
            (r1_min <= ts <= r1_max)
            or (r2_min <= ts <= r2_max)
            or (r3_min <= ts <= r3_max)
            or (r4_min <= ts <= r4_max)
        )

    @staticmethod
    def _period_day(ts: pd.Timestamp) -> str:
        """Return 'mañana' (05:00–11:59), 'tarde' (12:00–18:59), or 'noche' (19:00–04:59)."""
        if pd.isna(ts):
            return "noche"
        t = ts.time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("04:59", "%H:%M").time()

        if morning_min < t < morning_max:
            return "mañana"
        elif afternoon_min < t < afternoon_max:
            return "tarde"
        elif (evening_min < t < evening_max) or (night_min < t < night_max):
            return "noche"
        else:
            return "noche"

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Ensure all cols exist in df; create missing ones filled with 0."""
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        return df[cols]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Transform raw data into model-ready features (and target if requested).

        Args:
            data: Raw input DataFrame.
            target_column: If provided, also return the target as a DataFrame.

        Returns:
            (features, target) if target_column is provided, else features only.
        """
        df = data.copy()

        df["Fecha-I"] = pd.to_datetime(df["Fecha-I"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df["Fecha-O"] = pd.to_datetime(df["Fecha-O"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        if "high_season" not in df.columns:
            df["high_season"] = df["Fecha-I"].apply(self._is_high_season)
        if "min_diff" not in df.columns:
            df["min_diff"] = (df["Fecha-O"] - df["Fecha-I"]).dt.total_seconds() / 60.0
        if "period_day" not in df.columns:
            df["period_day"] = df["Fecha-I"].apply(self._period_day)

        if target_column:
            if target_column not in df.columns:
                df[target_column] = (df["min_diff"] > 15).astype(int)
            target = pd.DataFrame({target_column: df[target_column].astype(int)})

        if "MES" in df.columns:
            df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(-1).astype(int)

        opera = pd.get_dummies(df["OPERA"], prefix="OPERA")
        tipovuelo = pd.get_dummies(df["TIPOVUELO"], prefix="TIPOVUELO")
        mes = pd.get_dummies(df["MES"], prefix="MES")
        features = pd.concat([opera, tipovuelo, mes], axis=1)

        features = self._ensure_columns(features, self._TOP_FEATURES).astype(int)

        if target_column:
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train the model.

        Args:
            features: Preprocessed features.
            target: DataFrame containing the target column.
        """
        y = target.iloc[:, 0].astype(int)
        n_y1 = int((y == 1).sum())
        n_y0 = len(y) - n_y1
        scale = float(n_y0) / float(n_y1) if n_y1 > 0 else 1.0

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self._model.fit(features, y)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delay labels.

        Args:
            features: Preprocessed features.

        Returns:
            A list of integer predictions (0/1).
        """
        if self._model is None:
            return [0] * features.shape[0]
        preds = self._model.predict(features)
        return [int(v) for v in preds.tolist()]
