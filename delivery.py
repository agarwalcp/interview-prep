"""
Delivery analysis for SPOKEN answers (transcript-based). Separate from answer quality on purpose.

What is measured: filler words, speaking pace (approximate), repeated phrases.
What is NOT inferred: emotion, personality, "true" confidence. Camera motion and pitch spread are NOT used here
(they are not validated indicators of anything). Metrics without reliable evidence are reported as unavailable (None).
"""
import re
import scoring_config as C

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|,")
PREV_OK_BEFORE_LIKE = {'and', 'but', 'so', 'um', 'uh', 'just', 'then', 'because', 'that', 'which'}
NEXT_OK_AFTER_LIKE = {'i', 'we', 'you', 'it', 'they', 'the', 'my', 'our', 'there', 'this', 'a', 'an', 'when', 'if', 'really', 'very', 'kind', 'sort'}
YOUKNOW_LEGIT_PREV = {'do', 'did', "don't", 'dont', 'if', 'as', 'can', 'would', 'could', 'will', 'how', 'what', 'you'}
YOUKNOW_LEGIT_NEXT = {'that', 'how', 'what', 'why', 'when', 'where', 'which', 'who', 'it', 'about'}
IMEAN_LEGIT_PREV = {'what', "that's", 'thats', "here's", 'which', 'if'}


def _tok(text):
    return WORD.findall(str(text or '').lower())


def detect_fillers(text):
    """Returns (list of (filler, position)), total spoken words. Heuristic; errs on undercounting."""
    toks = _tok(text)
    words = [t for t in toks if t != ',']
    found = []
    always = set(C.ALWAYS_FILLERS)
    habitual = {w: sum(1 for t in toks if t == w) for w in C.HABITUAL_FILLERS}
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i > 0 else ''
        nxt = toks[i + 1] if i + 1 < len(toks) else ''
        if t in always:
            found.append(t)
        elif t in habitual and habitual[t] >= C.HABITUAL_MIN:
            found.append(t)
        elif t == 'like':
            comma_set_off = prev == ',' and nxt == ','
            if comma_set_off or (prev in PREV_OK_BEFORE_LIKE and nxt in NEXT_OK_AFTER_LIKE):
                found.append('like')
        elif t == 'you' and nxt == 'know':
            after = toks[i + 2] if i + 2 < len(toks) else ''
            if prev not in YOUKNOW_LEGIT_PREV and after not in YOUKNOW_LEGIT_NEXT:
                found.append('you know')
        elif t == 'i' and nxt == 'mean':
            after = toks[i + 2] if i + 2 < len(toks) else ''
            if prev not in IMEAN_LEGIT_PREV and after not in {'that', 'by', 'to'}:
                found.append('i mean')
    return found, len(words)


def repetition_ratio(text):
    words = [t for t in _tok(text) if t != ',']
    n = len(words)
    if n < 4:
        return 0.0
    immediate = sum(1 for i in range(1, n) if words[i] == words[i - 1] and len(words[i]) > 1)
    tri = {}
    for i in range(n - 2):
        g = tuple(words[i:i + 3])
        tri[g] = tri.get(g, 0) + 1
    repeated = sum(c - 1 for c in tri.values() if c > 1)
    return (immediate + repeated) / n


def _lin(x, zero, free):
    """1.0 at/below `free`, falling linearly to 0.0 at `zero`."""
    if x <= free:
        return 1.0
    return max(0.0, 1.0 - (x - free) / (zero - free))


def analyze_delivery(text, input_mode='typed', speech_seconds=None):
    """Returns a dict. Everything unmeasurable is None and listed in 'unavailable'."""
    if input_mode != 'spoken':
        return {'available': False, 'reason': 'Typed answer: speech delivery metrics do not apply.', 'score': None,
                'filler_count': None, 'filler_rate': None, 'words': None, 'components': {}, 'fillers': []}
    found, words = detect_fillers(text)
    out = {'available': False, 'words': words, 'filler_count': len(found), 'filler_rate': (len(found) / words * 100.0) if words else 0.0,
           'fillers': found, 'components': {}, 'unavailable': ['pauses (the browser speech API gives no pause timing)'], 'score': None,
           'note': 'Based on the speech-recognition transcript, which may omit or mishear fillers; pace is approximate.'}
    if words < C.MIN_WORDS_FOR_DELIVERY:
        out['reason'] = f'Too little speech ({words} words) for a delivery score.'
        return out
    comps = {'fillers': 100 * _lin(out['filler_rate'], C.FILLER_ZERO_RATE, C.FILLER_FREE_RATE),
             'repetition': 100 * _lin(repetition_ratio(text), C.REPEAT_ZERO_RATE, 0.0)}
    if speech_seconds and speech_seconds >= C.MIN_SECONDS_FOR_PACE:
        wpm = words / (speech_seconds / 60.0)
        lo, hi = C.IDEAL_WPM
        if lo <= wpm <= hi:
            pace = 1.0
        elif wpm < lo:
            pace = max(0.0, (wpm - C.PACE_ZERO_WPM[0]) / (lo - C.PACE_ZERO_WPM[0]))
        else:
            pace = max(0.0, (C.PACE_ZERO_WPM[1] - wpm) / (C.PACE_ZERO_WPM[1] - hi))
        comps['pace'] = 100 * pace
        out['wpm'] = wpm
    else:
        out['unavailable'].append('pace (recording time too short or not captured)')
    wsum = sum(C.DELIVERY_WEIGHTS[k] for k in comps)
    out['components'] = {k: round(v, 1) for k, v in comps.items()}
    out['score'] = sum(C.DELIVERY_WEIGHTS[k] * v for k, v in comps.items()) / wsum
    out['available'] = True
    out['reason'] = None
    return out


def delivery_feedback(d):
    if not d.get('available'):
        return [d.get('reason')] if d.get('reason') else []
    msgs = [f"You used {d['filler_count']} detected filler word{'s' if d['filler_count'] != 1 else ''} in {d['words']} spoken words "
            f"({d['filler_rate']:.1f}%)."]
    if d['filler_rate'] > C.FILLER_FREE_RATE * 3:
        msgs.append('Try pausing silently instead of filling the gap.')
    if d.get('wpm') is not None:
        lo, hi = C.IDEAL_WPM
        if d['wpm'] < lo: msgs.append(f"Estimated pace {d['wpm']:.0f} wpm is slower than the {lo:.0f}-{hi:.0f} wpm reference range (includes thinking time).")
        elif d['wpm'] > hi: msgs.append(f"Estimated pace {d['wpm']:.0f} wpm is faster than the {lo:.0f}-{hi:.0f} wpm reference range.")
    return msgs


def aggregate_delivery(per_answer):
    """per_answer: list of analyze_delivery() dicts for SPOKEN answers."""
    spoken = [d for d in per_answer if d and d.get('words') is not None]
    scored = [d for d in spoken if d.get('score') is not None]
    words = sum(d['words'] for d in spoken); fillers = sum(d['filler_count'] for d in spoken)
    if not spoken:
        return {'available': False, 'reason': 'No spoken answers: delivery metrics are unavailable for typed answers.',
                'score': None, 'filler_count': None, 'filler_rate': None, 'words': None, 'answers': 0}
    tw = sum(d['words'] for d in scored)
    return {'available': bool(scored), 'answers': len(spoken), 'words': words, 'filler_count': fillers,
            'filler_rate': (fillers / words * 100.0) if words else 0.0,
            'score': (sum(d['score'] * d['words'] for d in scored) / tw) if tw else None,
            'reason': None if scored else 'Spoken answers were too short for a delivery score.'}
