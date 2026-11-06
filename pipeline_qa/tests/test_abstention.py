import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_abstention import apply_policy, summarize, load_predictions


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Predicted_Answer': ['yes', 'no', '', 'incorrect', 'the'],
            'Confidence': [.9, .89, .99, .95, .99],
            'exact_match': [1, 0, 1, 0, 1],
            'answerable': [True, True, False, False, False],
            'model_abstained': [False, False, True, False, False]})

    def test_threshold_boundary_and_native_abstention(self):
        result = apply_policy(self.df, .9)
        self.assertEqual(result.answered.tolist(), [True, False, False, True, True])
        self.assertEqual(result.decision.iloc[2], 'astensione_modello')
        self.assertEqual(result.Final_Answer.iloc[1], 'NESSUNA RISPOSTA')
        self.assertEqual(self.df.Predicted_Answer.iloc[1], 'no')

    def test_risk_and_denominators(self):
        result = summarize(apply_policy(self.df, .9), .9, 'mixed')
        self.assertEqual(result['coverage'], 3/5)
        self.assertEqual(result['risk'], 2/3)
        self.assertEqual(result['abstention_rate_answerable'], 1/2)
        self.assertEqual(result['abstention_rate_unanswerable'], 1/3)

    def test_no_answers_means_undefined_risk(self):
        result = summarize(apply_policy(self.df, 1), 1, 'mixed')
        self.assertEqual(result['coverage'], 0)
        self.assertTrue(np.isnan(result['risk']))

    def test_baseline(self):
        result = summarize(apply_policy(self.df), None, 'mixed')
        self.assertEqual(result['n_answered'], 4)
        self.assertEqual(result['n_filter_abstained'], 0)

    def test_loading_preserves_na_and_removes_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.csv'
            pd.DataFrame({'ID': ['a', 'a', 'b'], 'Predicted_Answer': ['NA','NA','NESSUNA RISPOSTA'],
                          'Confidence': [.9,.9,.8], 'Exact_Match_Score': [1,1,0]}).to_csv(path,index=False)
            with self.assertWarns(UserWarning):
                result = load_predictions(path, True)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.model_abstained.tolist(), [False, True])


if __name__ == '__main__':
    unittest.main()
