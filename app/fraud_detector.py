"""Fraud detection using the recovered, calibrated preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PREPROCESSING_SPEC_PATH = MODEL_DIR / "preprocessing_spec.json"
RAW_SCALER_PATH = MODEL_DIR / "raw_scaler.pkl"
REVIEW_THRESHOLD = 0.01

_MODEL_PATTERN = "calibrated_fraud_model_2*.pkl"
_SCALER_PATTERN = "calibrated_fraud_scaler_2*.pkl"


def _latest_artifact(pattern: str) -> Path:
    matches = sorted(MODEL_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No model artifact matching {pattern!r} in {MODEL_DIR}")
    return matches[-1]


class FraudDetector:
    """Load and apply the recovered raw-scaler/calibrated-scaler/model pipeline."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        raw_scaler_path: str | Path = RAW_SCALER_PATH,
        calibrated_scaler_path: str | Path | None = None,
        spec_path: str | Path = PREPROCESSING_SPEC_PATH,
    ) -> None:
        self.available = False
        self.model = None
        self.raw_scaler = None
        self.calibrated_scaler = None
        self.spec: dict[str, Any] = {}

        try:
            self.spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
            self.type_map = self.spec["type_map"]
            self.feature_names = self.spec["feature_names"]
            self.ratio_eps = float(self.spec["ratio_eps"])
            self.model_path = Path(model_path) if model_path else _latest_artifact(_MODEL_PATTERN)
            self.calibrated_scaler_path = (
                Path(calibrated_scaler_path)
                if calibrated_scaler_path
                else _latest_artifact(_SCALER_PATTERN)
            )
            self.raw_scaler = joblib.load(raw_scaler_path)
            self.calibrated_scaler = joblib.load(self.calibrated_scaler_path)
            self.model = joblib.load(self.model_path)
            self.available = True
            print("✅ Fraud detector initialized successfully")
            print(f"   Model: {self.model.__class__.__name__}")
            print(f"   Features: {len(self.feature_names)}")
        except Exception as exc:
            print(f"⚠️ Error loading fraud pipeline: {exc}")

    def build_features(self, transaction: Mapping[str, Any]) -> np.ndarray:
        """Build the exact 11-column raw feature vector required by the pipeline."""
        if not isinstance(transaction, Mapping):
            raise TypeError("A transaction must be a mapping of transaction fields")
        transaction_type = transaction["type"]
        if transaction_type not in self.type_map:
            raise ValueError(
                f"Unknown transaction type {transaction_type!r}; "
                f"expected one of {sorted(self.type_map)}"
            )

        step = float(transaction["step"])
        amount = float(transaction["amount"])
        old_org = float(transaction["oldbalanceOrg"])
        new_org = float(transaction["newbalanceOrig"])
        old_dest = float(transaction["oldbalanceDest"])
        new_dest = float(transaction["newbalanceDest"])
        features = np.array([
            step, amount, old_org, new_org, old_dest, new_dest,
            new_org - old_org, new_dest - old_dest,
            amount / (old_org + self.ratio_eps),
            amount / (old_dest + self.ratio_eps),
            self.type_map[transaction_type],
        ], dtype=float)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict_matrix(self, features: np.ndarray) -> np.ndarray:
        transformed = self.raw_scaler.transform(features)
        transformed = self.calibrated_scaler.transform(transformed)
        return np.asarray(self.model.predict_proba(transformed)[:, 1], dtype=float)

    def predict(self, transaction: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> float | np.ndarray:
        """Return fraud probability for one transaction or an iterable batch."""
        if not self.available:
            raise RuntimeError("Fraud detection pipeline is not available")
        if isinstance(transaction, Mapping):
            return float(self._predict_matrix(self.build_features(transaction)[None, :])[0])
        return self._predict_matrix(np.vstack([self.build_features(tx) for tx in transaction]))

    def predict_batch(self, transactions: Iterable[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(self.predict(transactions), dtype=float)

    def check_transaction(
        self, amount: float, transaction_type: int | str, old_balance: float, new_balance: float
    ) -> dict[str, Any]:
        """Backward-compatible helper for the existing voice-agent commands."""
        if isinstance(transaction_type, int):
            transaction_type = {0: "PAYMENT", 1: "TRANSFER", 2: "CASH_OUT", 3: "CASH_IN"}.get(
                transaction_type, transaction_type
            )
        transaction = {
            "step": 1, "type": transaction_type, "amount": amount,
            "oldbalanceOrg": old_balance, "newbalanceOrig": new_balance,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        try:
            probability = float(self.predict(transaction))
            risk_level = ("🔴 HIGH RISK" if probability >= 0.8 else
                          "🟡 MEDIUM RISK" if probability >= 0.5 else "🟢 LOW RISK")
            is_fraud = probability >= REVIEW_THRESHOLD
            message = (f"⚠️ FRAUD ALERT! {probability:.2%} probability, {risk_level}"
                       if is_fraud else
                       f"✅ Transaction appears safe - {probability:.2%} probability, {risk_level}")
            return {"is_fraud": is_fraud, "probability": probability,
                    "risk_level": risk_level, "message": message}
        except Exception as exc:
            return {"is_fraud": False, "probability": 0.0, "risk_level": "ERROR",
                    "message": f"⚠️ Error: {exc}"}
