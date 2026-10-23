"""Option A grading: explainable keyword/concept matching (no black-box scoring)."""
import re

def _norm(t): return ' '.join(re.findall(r'[a-z0-9\-]+', t.lower()))

def grade_answer(answer, expected_keywords):
    kws = [k.strip().lower() for k in str(expected_keywords).split(',') if k.strip() and k.strip() != 'nan']
    text = _norm(answer)
    if not kws:
        return {'score': 0.0, 'matched': [], 'missing': [], 'correct': False}
    matched = [k for k in kws if k in text or (len(k) >= 6 and k[:5] in text)]
    missing = [k for k in kws if k not in matched]
    score = len(matched) / len(kws)
    return {'score': score, 'matched': matched, 'missing': missing,
            'correct': score >= 0.5 and len(matched) >= 1}

def hr_substance(answer):
    """HR answers have no 'correct' answer; use length of the answer (capped) as a crude substance proxy."""
    return min(1.0, len(answer.split()) / 40.0)
