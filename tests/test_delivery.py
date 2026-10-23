import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import delivery as D
import scoring_config as C
from scoring import Scorer
import pandas as pd
from paths import QUESTION_BANK

CLEAN = 'A closure is a function that remembers the variables of the scope where it was created, even after the outer function has returned, which is useful for private state.'
UMS = 'Um a closure is, uh, a function that um remembers the variables of the scope where it was created, uh, even after the outer function has returned, um which is useful.'


class DeliveryCases(unittest.TestCase):
    def test_01_no_fillers(self):
        d = D.analyze_delivery(CLEAN, 'spoken', 20); self.assertEqual(d['filler_count'], 0); self.assertEqual(d['filler_rate'], 0.0)

    def test_02_multiple_fillers_count_and_rate(self):
        d = D.analyze_delivery(UMS, 'spoken', 20); words = len([w for w in UMS.lower().replace(',', ' ').split()])
        self.assertEqual(d['filler_count'], 5); self.assertAlmostEqual(d['filler_rate'], 5 / words * 100, places=6)

    def test_03_legitimate_like_not_filler(self):
        for t in ['I like pizza and I would like to learn more about how a hash map works like a dictionary in practice today.',
                  'It looks like this approach is something like that which many people like to use in their own projects.']:
            self.assertEqual(D.detect_fillers(t)[0], [], t)
        self.assertIn('like', D.detect_fillers('so the function, like, returns another function and like the inner one keeps scope')[0])

    def test_you_know_and_i_mean_context(self):
        self.assertEqual(D.detect_fillers('Do you know what a closure is? It is what I mean by scope here.')[0], [])
        self.assertEqual(D.detect_fillers('It stores state, you know, and it is private')[0], ['you know'])

    def test_04_empty_transcript_no_division_error(self):
        for t in ['', '   ', None]:
            d = D.analyze_delivery(t, 'spoken', 5); self.assertEqual(d['filler_rate'], 0.0); self.assertIsNone(d['score']); self.assertFalse(d['available'])

    def test_05_typed_answer_speech_metrics_unavailable(self):
        d = D.analyze_delivery(UMS, 'typed'); self.assertFalse(d['available']); self.assertIsNone(d['filler_count']); self.assertIsNone(d['score'])

    def test_single_um_not_penalised_heavily(self):
        t = 'um ' + CLEAN + ' ' + CLEAN
        d = D.analyze_delivery(t, 'spoken', 30); self.assertEqual(d['filler_count'], 1); self.assertGreaterEqual(d['components']['fillers'], 90)

    def test_filler_penalty_is_proportional(self):
        few, many = D.analyze_delivery(UMS, 'spoken', 20), D.analyze_delivery('um uh um uh ' * 6 + CLEAN, 'spoken', 20)
        self.assertGreater(few['score'], many['score'])

    def test_pace_unavailable_without_time_and_filler_does_not_dominate(self):
        d = D.analyze_delivery(CLEAN, 'spoken', None); self.assertNotIn('pace', d['components']); self.assertTrue(d['available'])
        self.assertTrue(any('pace' in u for u in d['unavailable']))

    def test_06_07_quality_and_delivery_are_independent(self):
        sc = Scorer(pd.read_csv(QUESTION_BANK)); good = ('Overfitting occurs when a model learns the training data too closely, including noise, and performs poorly on unseen data. '
              'Underfitting is when the model is too simple. To prevent overfitting you can use regularization, dropout, early stopping or more data.')
        q = {'question': 'Explain Overfitting and Underfitting. How can you prevent them?', 'role': 'Data Scientist', 'expected_keywords': 'overfitting'}
        noisy = 'um ' + good.replace('. ', ' uh. ') + ' um'
        qa, qb = sc.evaluate_technical(q, good)['quality'], sc.evaluate_technical(q, noisy)['quality']
        self.assertGreaterEqual(qb, qa - 5)                                            # fillers alone do not drop the technical score
        self.assertGreater(D.analyze_delivery(good, 'spoken', 25)['score'], D.analyze_delivery(noisy, 'spoken', 25)['score'])   # but delivery reflects them
        fluent_wrong = 'Overfitting is when the model is very accurate on all data and this is always a good thing for every model that we build today.'
        self.assertLess(sc.evaluate_technical(q, fluent_wrong)['quality'], 30)
        self.assertGreaterEqual(D.analyze_delivery(fluent_wrong, 'spoken', 12)['score'], 90)

    def test_aggregate(self):
        a = D.aggregate_delivery([D.analyze_delivery(UMS, 'spoken', 20), D.analyze_delivery('x', 'typed')])
        self.assertEqual(a['answers'], 1); self.assertEqual(a['filler_count'], 5)
        self.assertFalse(D.aggregate_delivery([D.analyze_delivery('x', 'typed')])['available'])


if __name__ == '__main__':
    unittest.main()
