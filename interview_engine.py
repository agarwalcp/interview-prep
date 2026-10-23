"""
Headless interview engine: question selection, grading, Bayesian skill update,
Markov state tracking, RBM confidence. The Streamlit UI (app.py) is a thin layer on top.

Routing rule: technical answers -> keyword grading -> Bayesian Network + Markov chain.
              HR answers        -> NO correctness grading -> RBM confidence only.
"""
import random
import numpy as np
import pandas as pd
from skill_network import ROLE_SKILL_GRAPHS, RoleSkillNetwork, InterviewMarkovChain
from scoring import Scorer, estimate_difficulty
from delivery import analyze_delivery, delivery_feedback, aggregate_delivery
import scoring_config as C
from confidence import ConfidenceModel

FAST_LATENCY_SECONDS = 60.0   # only used by the experimental RBM latent score
INTERVIEW_TYPES = ('Technical', 'HR/Behavioral', 'Mixed')
DIFFICULTIES = ('Beginner', 'Intermediate', 'Advanced')
MINUTES_PER_TECH, MINUTES_PER_HR = 3, 2
_DIFF_ORDER = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
_SCORERS = {}


def get_scorer(bank):
    key = id(bank)
    if key not in _SCORERS:
        _SCORERS[key] = Scorer(bank)
    return _SCORERS[key]


def sensor_features(sensors, defaults):
    """Raw sensor readings -> the two RBM behaviour inputs (neutral default if sensor off/unusable).
    EXPERIMENTAL: used only by the RBM latent score, never by the answer or delivery scores."""
    calm, steady = float(defaults[2]), float(defaults[3])
    if sensors:
        if sensors.get('motion_mean') is not None:
            calm = 1.0 - min(1.0, sensors['motion_mean'] * 5.0)
        if sensors.get('pitch_std') is not None and (sensors.get('voiced_ratio') or 0) >= 0.1:
            steady = 1.0 - min(1.0, sensors['pitch_std'] / 100.0)
    return calm, steady


def estimated_minutes(n_tech, n_hr):
    return n_tech * MINUTES_PER_TECH + n_hr * MINUTES_PER_HR


class InterviewSession:
    def __init__(self, role, question_bank, n_tech=6, n_hr=2, seed=None, confidence_model=None,
                 interview_type='Mixed', difficulty='Intermediate', candidate_name=''):
        if interview_type == 'Technical': n_hr = 0
        if interview_type == 'HR/Behavioral': n_tech = 0
        self.role, self.interview_type, self.difficulty = role, interview_type, difficulty
        self.candidate_name = str(candidate_name or '')[:40]
        self.rng = random.Random(seed)
        self.cfg = ROLE_SKILL_GRAPHS[role]
        self.net = RoleSkillNetwork(self.cfg['edges'], self.cfg['nodes'])
        self.chain = InterviewMarkovChain()
        self.conf = confidence_model or ConfidenceModel()
        self.scorer = get_scorer(question_bank)
        self.difficulty_note = None
        self.questions = self._select(question_bank, n_tech, n_hr)
        self.records = []

    # ---- question selection: round-robin over skill nodes (whole graph gets evidence), preferring the requested difficulty ----
    def _select(self, qb, n_tech, n_hr):
        chosen = []
        if n_tech > 0:
            tech = qb[(qb.role == self.role) & (~qb.is_hr)].copy()
            tech['skill'] = tech.category.map(self.cfg['category_map'])
            tech = tech.dropna(subset=['skill'])
            tech = tech[tech.expected_keywords.fillna('').str.strip() != '']
            tech['difficulty'] = tech.question.map(estimate_difficulty)
            want = _DIFF_ORDER[self.difficulty]
            by_skill = {}
            for sk, g in tech.groupby('skill'):
                rows = g.to_dict('records'); self.rng.shuffle(rows)
                rows.sort(key=lambda r: -abs(_DIFF_ORDER[r['difficulty']] - want))     # best match last (we pop from the end)
                by_skill[sk] = rows
            skills = list(by_skill); self.rng.shuffle(skills)
            while len(chosen) < n_tech and any(by_skill.values()):
                for s in skills:
                    if by_skill[s] and len(chosen) < n_tech:
                        chosen.append(by_skill[s].pop())
            off = sum(1 for q in chosen if q['difficulty'] != self.difficulty)
            if off:
                self.difficulty_note = (f'{off} of {len(chosen)} technical questions are not rated {self.difficulty} '
                                        f'(not enough {self.difficulty} questions for this role). Difficulty is a wording-based estimate.')
        hr = qb[qb.is_hr].to_dict('records'); self.rng.shuffle(hr)
        return chosen + [dict(q, skill=None, difficulty=None) for q in hr[:n_hr]]

    # ---- one answer ----
    def submit(self, idx, answer, latency_seconds=0.0, sensors=None, input_mode='typed', speech_seconds=None):
        q = self.questions[idx]
        if q['is_hr']:
            ev = self.scorer.evaluate_hr(q, answer)
        else:
            ev = self.scorer.evaluate_technical(q, answer, q.get('difficulty'))
            self.chain.step(bool(ev['correct']))
        delivery = analyze_delivery(answer, input_mode, speech_seconds)
        # experimental latent score (RBM trained on synthetic profiles) - kept for the syllabus demo, NOT part of any headline score
        calm, steady = sensor_features(sensors, self.conf.defaults)
        fast = float(np.clip(1.0 - (latency_seconds or 0.0) / FAST_LATENCY_SECONDS, 0, 1)) * ev['score']
        feats = [ev['score'], fast, calm, steady]
        rec = {'index': idx, 'question': q['question'], 'is_hr': bool(q['is_hr']), 'skill': q.get('skill'),
               'category': q.get('category'), 'difficulty': q.get('difficulty'), 'answer': answer, 'input_mode': input_mode,
               **ev, 'delivery': delivery, 'delivery_feedback': delivery_feedback(delivery),
               'features': feats, 'experimental_latent': float(self.conf.score(feats)[0])}
        ref = q.get('reference')
        rec['reference'] = ref if isinstance(ref, str) and ref.strip() else None
        self.records.append(rec)
        return rec

    # ---- report ----
    def skill_beliefs(self):
        per_skill = {}
        for r in self.records:
            if not r['is_hr'] and r['skill']:
                per_skill.setdefault(r['skill'], []).append(r['quality'])
        evidence = {s: int(np.mean(v) >= C.PASS_THRESHOLD) for s, v in per_skill.items()}
        return self.net.beliefs(evidence), evidence

    def what_if(self, horizon=5):
        """Forward-looking estimate: if the candidate mastered weak skill X, how would the skill map and the chance of being in
        the 'Confident' interview state `horizon` answers from now change? Model-based (hand-set priors; belief used as
        P(correct)) - an estimate, not a measurement."""
        beliefs, evidence = self.skill_beliefs()
        base_p = float(np.mean(list(beliefs.values())))
        start = self.chain.state
        base = InterviewMarkovChain.project(base_p, horizon, start)['Confident']
        out = []
        for s, b in beliefs.items():
            if b >= 0.5:
                continue
            ev = dict(evidence); ev[s] = 1
            new_p = float(np.mean(list(self.net.beliefs(ev).values())))
            conf = InterviewMarkovChain.project(new_p, horizon, start)['Confident']
            out.append({'skill': s, 'belief_now': b, 'avg_skill_before': base_p, 'avg_skill_after': new_p,
                        'confident_before': base, 'confident_after': conf})
        out.sort(key=lambda x: x['confident_after'] - x['confident_before'], reverse=True)
        return {'horizon': horizon, 'start_state': start, 'items': out}

    def report(self):
        beliefs, evidence = self.skill_beliefs()
        weak = sorted([s for s, b in beliefs.items() if b < 0.5], key=lambda s: beliefs[s])
        tech = [r for r in self.records if not r['is_hr']]
        hr = [r for r in self.records if r['is_hr']]
        mean = lambda xs: float(np.mean(xs)) if xs else None
        dim_avg = {k: mean([r['dimensions'][k] for r in tech]) for k in C.TECH_WEIGHTS}
        hr_avg = {k: mean([r['dimensions'][k] for r in hr]) for k in C.HR_WEIGHTS}
        delivery = aggregate_delivery([r['delivery'] for r in self.records])
        strengths, improvements = [], []
        for r in sorted(self.records, key=lambda r: -r['quality'])[:2]:
            if r['quality'] >= C.PASS_THRESHOLD and r['feedback']['strengths']:
                strengths.append(f"Q{r['index'] + 1}: {r['feedback']['strengths'][0]}")
        for r in sorted(self.records, key=lambda r: r['quality'])[:3]:
            if r['quality'] < C.PASS_THRESHOLD and r['feedback']['improvements']:
                improvements.append(f"Q{r['index'] + 1}: {r['feedback']['improvements'][0]}")
        if delivery['available'] and delivery['filler_rate'] > C.FILLER_FREE_RATE * 3:
            improvements.append(f"Delivery: {delivery['filler_count']} detected filler words in {delivery['words']} spoken words ({delivery['filler_rate']:.1f}%).")
        revise = list(dict.fromkeys([s.replace('_', ' ') for s in weak] + [m for r in tech if r['quality'] < C.PASS_THRESHOLD for m in r['missing'][:2]]))[:8]
        return {'overall_score': mean([r['quality'] for r in self.records]),
                'technical_quality': mean([r['quality'] for r in tech]), 'hr_quality': mean([r['quality'] for r in hr]),
                'dimension_averages': dim_avg, 'hr_dimension_averages': hr_avg, 'delivery': delivery,
                'answered': len(self.records), 'total': len(self.questions),
                'strengths': strengths, 'improvements': improvements, 'revise_topics': revise,
                'experimental_latent_score': mean([r['experimental_latent'] for r in self.records]),
                'beliefs': beliefs, 'evidence': evidence, 'weak_skills': weak, 'state_history': self.chain.history,
                'records': self.records, 'difficulty_note': self.difficulty_note,
                'low_evidence_count': sum(1 for r in tech if 'low_reference_evidence' in r['flags'])}
