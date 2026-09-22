"""Small checks for the mistakes that would change the conclusions."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from analysis_helpers import make_features, evaluate, threshold_comparison


class CoreChecks(unittest.TestCase):
    def test_only_initiation_inputs(self):
        raw = pd.DataFrame({'amount': [0, 100], 'type': ['CASH_OUT', 'TRANSFER']})
        self.assertEqual(list(make_features(raw)), ['log_amount', 'is_transfer'])
        for field in ['isFraud', 'isFlaggedFraud', 'oldbalanceOrg', 'newbalanceOrig', 'step']:
            with self.assertRaises(ValueError):
                make_features(raw.assign(**{field: 1}))

    def test_metrics_match_sklearn(self):
        labels = [0, 0, 1, 1, 1]
        scores = np.array([0.1, 0.8, 0.2, 0.7, 0.9])
        result = evaluate(labels, scores, 0.7)
        predicted = scores >= 0.7
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertAlmostEqual(result['precision'], precision_score(labels, predicted))
        self.assertAlmostEqual(result['recall'], recall_score(labels, predicted))
        self.assertAlmostEqual(result['F1'], f1_score(labels, predicted))

    def test_zero_alerts_are_not_perfect_precision(self):
        result = evaluate([0, 1], [0.1, 0.2], 0.5)
        self.assertTrue(np.isnan(result['precision']))
        self.assertEqual(result['recall'], 0)

    def test_threshold_direction_and_equality(self):
        table = threshold_comparison([0, 1, 1], [0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
        self.assertEqual(table['alerts'].tolist(), [3, 2, 1])


if __name__ == '__main__':
    unittest.main()
