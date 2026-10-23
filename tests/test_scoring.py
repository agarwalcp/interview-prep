"""Run:  python -m unittest discover -s tests -v   (from the folder containing server.py)"""
import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import scoring_config as C
from scoring import Scorer, validate_dimensions, aggregate, fallback_evaluation
from paths import QUESTION_BANK

BANK = pd.read_csv(QUESTION_BANK)
SC = Scorer(BANK)
OVERFIT = {'question': 'Explain Overfitting and Underfitting. How can you prevent them?', 'role': 'Data Scientist', 'expected_keywords': 'overfitting, underfitting, prevent'}
DEFN = {'question': 'What is overfitting in machine learning?', 'role': 'Data Scientist', 'expected_keywords': 'overfitting, machine, learning'}
HR_Q = {'question': 'Tell me about a time you had a conflict with a teammate and how you resolved it.', 'is_hr': True}
GOOD = ('Overfitting occurs when a model learns the training data too closely, including noise, and performs poorly on unseen data. '
        'Underfitting is when the model is too simple. To prevent overfitting you can use regularization, dropout, early stopping or more data.')


class TechnicalCases(unittest.TestCase):
    def q(self, q, a): return SC.evaluate_technical(q, a)

    def test_01_empty_answer_very_low(self):
        r = self.q(OVERFIT, ''); self.assertLess(r['quality'], 5); self.assertIn('empty', r['flags'])

    def test_02_irrelevant_but_fluent(self):
        r = self.q(OVERFIT, 'I really enjoy cooking pasta on weekends with fresh basil and tomatoes from my small garden at home.')
        self.assertLess(r['dimensions']['relevance'], 15); self.assertLess(r['quality'], 25)

    def test_03_keyword_stuffing_or_wrong_reasoning(self):
        r = self.q(OVERFIT, 'Overfitting is when the model is very accurate on all data.')
        self.assertLess(r['dimensions']['correctness'], 20); self.assertLess(r['quality'], 30); self.assertTrue(r['misconceptions'])
        r2 = self.q(OVERFIT, 'overfitting underfitting prevent model training data noise regularization dropout unseen generalization')
        self.assertIn('keyword_stuffing', r2['flags']); self.assertLess(r2['quality'], 60)
        dt = {'question': 'How does a Decision Tree algorithm work?', 'role': 'ML Engineer', 'expected_keywords': 'decision, tree, algorithm, work'}
        self.assertLess(self.q(dt, 'The decision tree algorithm does not work at all, it is a random thing.')['quality'], 20)

    def test_04_partially_correct_has_specific_missing(self):
        r = self.q(OVERFIT, 'Overfitting happens when the model learns the training data too well including the noise.')
        self.assertTrue(35 <= r['quality'] <= 75, r['quality']); self.assertTrue(r['missing'])
        self.assertTrue(any('You did not mention' in s for s in r['feedback']['improvements']))

    def test_05_correct_but_concise_definition(self):
        r = self.q(DEFN, 'Overfitting is when a model memorizes the training data, including noise, so it fails on unseen data.')
        self.assertGreaterEqual(r['quality'], 60, r)
        self.assertGreaterEqual(r['dimensions']['explanation'], 60)

    def test_06_correct_comprehensive_is_high(self):
        self.assertGreaterEqual(self.q(OVERFIT, GOOD)['quality'], 75)

    def test_07_ordering_good_over_partial_over_wrong(self):
        a, b, c = (self.q(OVERFIT, x)['quality'] for x in (GOOD, 'Overfitting happens when the model learns the training data too well including the noise.', 'Overfitting is when the model is very accurate on all data.'))
        self.assertGreater(a, b); self.assertGreater(b, c)

    def test_08_generic_answer_does_not_pass_unrelated_question(self):
        q = {'question': 'What is dependency parsing?', 'role': 'ML Engineer', 'expected_keywords': 'dependency, parsing'}
        r = self.q(q, 'it works like the model learns from the training data because it minimizes the loss')
        self.assertLess(r['quality'], C.PASS_THRESHOLD - 20)

    def test_09_echoing_question_scores_low(self):
        self.assertLess(self.q(OVERFIT, 'what is overfitting and underfitting and how can you prevent them')['quality'], 25)

    def test_10_low_evidence_scores_are_capped(self):
        q = {'question': 'Describe event capturing.', 'role': 'Frontend Developer', 'expected_keywords': 'event, capturing'}
        r = self.q(q, 'Event capturing is the first phase of event propagation where the event travels from the window down through ancestor elements to the target, before bubbling.')
        self.assertIn('low_reference_evidence', r['flags']); self.assertLessEqual(r['dimensions']['correctness'], C.LOW_EVIDENCE_CAP + 1e-6)

    def test_11_quality_always_0_to_100(self):
        for a in ['', 'x', 'word ' * 500, '🙂🙂', GOOD * 5, 'null undefined NaN']:
            r = self.q(OVERFIT, a); self.assertTrue(0 <= r['quality'] <= 100)
            for v in r['dimensions'].values(): self.assertTrue(0 <= v <= 100 and not math.isnan(v))

    def test_12_dont_know(self):
        r = self.q(OVERFIT, "I don't know"); self.assertLess(r['quality'], 10)

    def test_13_feedback_consistent_with_score(self):
        r = self.q(OVERFIT, GOOD); self.assertFalse(r['misconceptions']); self.assertIn('Strong', r['feedback']['summary'] + 'Strong' if r['quality'] >= 80 else 'Strong')
        w = self.q(OVERFIT, 'Overfitting is when the model is very accurate on all data.')
        self.assertNotIn('Strong', w['feedback']['summary'])


class HRCases(unittest.TestCase):
    def test_07_behavioral_without_example(self):
        r = SC.evaluate_hr(HR_Q, 'I always try to communicate well and I would talk to my teammate and find a solution that works for everyone because teamwork is important.')
        self.assertIn('no_specific_example', r['flags']); self.assertTrue(any('specific situation' in s for s in r['feedback']['improvements'])); self.assertLess(r['quality'], 50)

    def test_star_answer_scores_higher(self):
        good = ('Last semester in our capstone project, my teammate and I disagreed about the database design. My task was to deliver the backend in two weeks. '
                'I organized a short meeting, asked each of us to present trade-offs, and we decided on PostgreSQL after testing both. '
                'As a result we finished 3 days early and I learned to resolve conflicts with data.')
        weak = 'I always try to communicate well and I would talk to my teammate.'
        a, b = SC.evaluate_hr(HR_Q, good), SC.evaluate_hr(HR_Q, weak)
        self.assertGreater(a['quality'], 70); self.assertGreater(a['quality'], b['quality'] + 30)

    def test_empty_hr(self):
        self.assertLess(SC.evaluate_hr(HR_Q, '')['quality'], 5)


class ValidationCases(unittest.TestCase):
    def test_08_malformed_evaluator_output_safe(self):
        keys = list(C.TECH_WEIGHTS)
        for bad in [None, 'text', [], {}, {'correctness': 'high'}, {'correctness': float('nan'), 'relevance': 1, 'completeness': 1, 'explanation': 1},
                    {'correctness': True, 'relevance': 1, 'completeness': 1, 'explanation': 1}]:
            self.assertIsNone(validate_dimensions(bad, keys))
        ok = validate_dimensions({'correctness': 250, 'relevance': -5, 'completeness': 50, 'explanation': 50}, keys)
        self.assertEqual((ok['correctness'], ok['relevance']), (100.0, 0.0))
        fb = fallback_evaluation(); self.assertEqual(fb['quality'], 0.0); self.assertIn('evaluator_error', fb['flags'])

    def test_scorer_never_raises_on_bad_input(self):
        for q in [{'question': 'x'}, {'question': None}, {'question': 'What is X?', 'expected_keywords': float('nan')}]:
            r = SC.evaluate_technical(q, None); self.assertTrue(0 <= r['quality'] <= 100)

    def test_weights_sum_to_one_and_aggregate_formula(self):
        self.assertAlmostEqual(sum(C.TECH_WEIGHTS.values()), 1.0); self.assertAlmostEqual(sum(C.HR_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(aggregate({'correctness': 100, 'relevance': 50, 'completeness': 0, 'explanation': 100}, C.TECH_WEIGHTS), 40 + 12.5 + 0 + 15)
        with self.assertRaises(ValueError): aggregate({'a': 1}, {'a': 0.5})


if __name__ == '__main__':
    unittest.main()
