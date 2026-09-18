"""
Fraud Detection Module for Voice Agent
Uses trained XGBoost model with StandardScaler
"""
import joblib
import numpy as np
import os

class FraudDetector:
    def __init__(self, model_path='./models/xgboost_model_local.pkl',
                 scaler_path='./models/scaler_local.pkl'):
        """
        Initialize the fraud detector with trained model and scaler
        """
        self.available = False
        self.model = None
        self.scaler = None

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.available = True
                print("✅ Fraud detector initialized successfully")
                print(f"   Model: {self.model.__class__.__name__}")
                print(f"   Features: {self.model.n_features_in_}")
            except Exception as e:
                print(f"⚠️ Error loading fraud model: {e}")
        else:
            print("⚠️ Fraud model files not found. Please train the model first.")

    def check_transaction(self, amount, transaction_type, old_balance, new_balance):
        """
        Check if a transaction is fraudulent

        Args:
            amount (float): Transaction amount
            transaction_type (int): 0=PAYMENT, 1=TRANSFER, 2=CASH_OUT, 3=CASH_IN
            old_balance (float): Sender's balance before transaction
            new_balance (float): Sender's balance after transaction

        Returns:
            dict: {
                'is_fraud': bool,
                'probability': float,
                'risk_level': str,
                'message': str
            }
        """
        if not self.available:
            return {
                "is_fraud": False,
                "probability": 0.0,
                "risk_level": "UNAVAILABLE",
                "message": "⚠️ Fraud detection unavailable - model not loaded"
            }

        try:
            # Generate realistic destination balances based on transaction type
            if transaction_type == 1:  # TRANSFER
                dest_old = np.random.uniform(100, 50000)
                dest_new = dest_old + amount
            elif transaction_type == 2:  # CASH_OUT
                dest_old = 0
                dest_new = 0
            else:  # PAYMENT, CASH_IN
                dest_old = np.random.uniform(100, 10000)
                dest_new = dest_old

            # Create feature vector (11 features matching training data)
            raw_features = np.array([[
                1,                          # step (time)
                amount,                     # amount
                old_balance,                # oldbalanceOrg
                new_balance,                # newbalanceOrig
                dest_old,                   # oldbalanceDest
                dest_new,                   # newbalanceDest
                new_balance - old_balance,  # balance_change_org
                dest_new - dest_old,        # balance_change_dest
                amount / (old_balance + 1), # amount_ratio_org
                amount / (dest_old + 1) if dest_old > 0 else 0,  # amount_ratio_dest
                transaction_type            # type_encoded
            ]])

            # Standardize features using saved scaler
            standardized = self.scaler.transform(raw_features)

            # Predict
            is_fraud = bool(self.model.predict(standardized)[0])
            probability = float(self.model.predict_proba(standardized)[0][1])

            # Determine risk level
            if probability >= 0.8:
                risk_level = "🔴 HIGH RISK"
            elif probability >= 0.5:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"

            # Generate message
            if is_fraud:
                message = f"⚠️ FRAUD ALERT! {probability:.2%} probability, {risk_level}"
            else:
                message = f"✅ Transaction appears safe - {probability:.2%} probability, {risk_level}"

            return {
                "is_fraud": is_fraud,
                "probability": probability,
                "risk_level": risk_level,
                "message": message
            }

        except Exception as e:
            print(f"❌ Error checking transaction: {e}")
            return {
                "is_fraud": False,
                "probability": 0.0,
                "risk_level": "ERROR",
                "message": f"⚠️ Error: {str(e)}"
            }

# Quick test
if __name__ == "__main__":
    print("🔍 Testing Fraud Detector")
    print("="*50)

    detector = FraudDetector()

    if detector.available:
        print("\n📊 Test Cases:")
        print("-"*40)

        test_cases = [
            (100, 0, 1000, 900, "Normal payment"),
            (5000, 1, 100, 0, "Fraudulent transfer"),
            (200, 2, 5000, 4800, "ATM withdrawal"),
            (9500, 2, 5000, 0, "Large cash out"),
            (50, 1, 2000, 1950, "Small transfer"),
        ]

        for amount, ttype, old_bal, new_bal, desc in test_cases:
            result = detector.check_transaction(amount, ttype, old_bal, new_bal)
            status = "⚠️ FRAUD" if result["is_fraud"] else "✅ Safe"
            print(f"{desc:<25}: {status} ({result['probability']:.2%})")
