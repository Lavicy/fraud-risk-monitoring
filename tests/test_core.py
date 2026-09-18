"""Targeted checks of leakage boundaries, chronology and metric semantics."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from fraud_utils import (RAW_FEATURES, make_features, features_for, make_model, fit_model,
                         choose_threshold, metrics, time_windows, distribution_bins, population_stability)


class CoreLogicTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({'type': ['TRANSFER', 'CASH_OUT', 'TRANSFER', 'CASH_OUT'],
                                 'amount': [100.0, 10.0, 0.0, 200.0],
                                 'oldbalanceOrg': [100.0, 0.0, 0.0, 300.0]})

    def test_prohibited_inputs_rejected(self):
        for column in ['isFraud', 'isFlaggedFraud', 'step', 'newbalanceOrig', 'newbalanceDest',
                       'origin_balance_error', 'nameOrig']:
            with self.subTest(column=column), self.assertRaises(ValueError):
                make_features(self.raw.assign(**{column: 1}))
        features = make_features(self.raw)
        self.assertEqual(features.requests_full_balance.tolist(), [1, 0, 0, 0])
        self.assertTrue(np.isfinite(features.drop(columns='type')).all().all())

    def test_balance_free_ablation_has_no_balance_dependency(self):
        before = features_for(self.raw, 'amount_and_type_only')
        after = features_for(self.raw.assign(oldbalanceOrg=1e12), 'amount_and_type_only')
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(list(before), ['type', 'log_amount'])

    def test_whole_step_windows_are_disjoint(self):
        data = pd.DataFrame({'step': [1, 2, 2, 3, 4, 5]})
        first, second, third = time_windows(data, [0, 2, 4, 5])
        self.assertEqual(first.step.tolist(), [1, 2, 2])
        self.assertEqual(len(first) + len(second) + len(third), len(data))
        self.assertTrue(first.index.intersection(second.index).empty)
        with self.assertRaises(ValueError):
            time_windows(data, [0, 3, 2, 5])

    def test_metrics_match_independent_sklearn_calculations(self):
        target = np.array([0, 1, 1, 0, 1, 0])
        scores = np.array([.1, .8, .4, .8, .8, .2])
        predicted = scores >= .8
        result = metrics(target, scores, .8)
        self.assertEqual((result['TP'], result['FP'], result['FN'], result['TN']), (2, 1, 1, 2))
        for key, expected in [('precision', precision_score(target, predicted)),
                              ('recall', recall_score(target, predicted)), ('F1', f1_score(target, predicted)),
                              ('AP', average_precision_score(target, scores))]:
            self.assertAlmostEqual(result[key], expected)
        self.assertEqual(result['alerts'], int(predicted.sum()))

    def test_threshold_search_including_ties_matches_brute_force(self):
        target = np.array([0, 1, 0, 1, 1, 0, 1])
        scores = np.array([.1, .7, .7, .9, .9, .2, .3])
        threshold, _ = choose_threshold(target, scores)
        options = [metrics(target, scores, float(t)) for t in np.unique(scores)]
        best = max(options, key=lambda row: (row['F1'], row['precision'], row['threshold']))
        self.assertEqual(threshold, best['threshold'])

    def test_undefined_rates_and_single_class_ap(self):
        result = metrics([0, 0], [.1, .2], .5)
        self.assertTrue(np.isnan(result['precision']))
        self.assertTrue(np.isnan(result['recall']))
        self.assertTrue(np.isnan(result['AP']))
        empty = metrics(np.array([], dtype=int), np.array([]), .5)
        self.assertEqual(empty['transactions'], 0)
        self.assertTrue(np.isnan(empty['alert_rate']))

    def test_preprocessing_is_unchanged_by_later_scoring(self):
        model = fit_model(make_model('logistic'), features_for(self.raw), [1, 0, 0, 1])
        scaler = model.named_steps['preprocess'].named_transformers_['numeric']
        before = scaler.mean_.copy()
        model.predict_proba(features_for(self.raw.assign(amount=1e8)))
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertEqual(scaler.n_samples_seen_, len(self.raw))

    def test_reference_bins_and_identical_distribution(self):
        reference = np.array([0, 0, 0, 1, 2, 10], dtype=float)
        edges = distribution_bins(reference)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertEqual(population_stability(reference, reference, edges), 0)
        self.assertTrue(np.isnan(population_stability(reference, [], edges)))


if __name__ == '__main__':
    unittest.main()
