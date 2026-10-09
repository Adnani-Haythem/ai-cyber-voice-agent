import pytest

from app.fraud_detector import FraudDetector, REVIEW_THRESHOLD


TRANSACTIONS = [
    (
        {
            "step": 10,
            "type": "PAYMENT",
            "amount": 120.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4880.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
        0.0,
        False,
    ),
    (
        {
            "step": 10,
            "type": "CASH_OUT",
            "amount": 9500.0,
            "oldbalanceOrg": 9500.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 9500.0,
        },
        0.848,
        True,
    ),
    (
        {
            "step": 10,
            "type": "TRANSFER",
            "amount": 181000.0,
            "oldbalanceOrg": 181000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
        0.850,
        True,
    ),
]


def test_recovered_pipeline_scores_verified_transactions():
    detector = FraudDetector()
    assert detector.available

    probabilities = detector.predict_batch([transaction for transaction, _, _ in TRANSACTIONS])
    for (_, expected, should_review), probability in zip(TRANSACTIONS, probabilities):
        print(f"p(fraud)={probability:.6f} review={probability >= REVIEW_THRESHOLD}")
        assert abs(probability - expected) <= 0.01
        assert bool(probability >= REVIEW_THRESHOLD) is should_review

def test_unknown_transaction_type_raises_value_error():
    detector = FraudDetector()
    transaction = dict(TRANSACTIONS[0][0], type="UNKNOWN")
    with pytest.raises(ValueError, match="Unknown transaction type"):
        detector.build_features(transaction)
