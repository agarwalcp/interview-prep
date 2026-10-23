"""
Answer evaluation: a hybrid, local, explainable evaluator (no LLM, no paid API).

Pipeline per technical answer
  1. rule checks (empty / too short / "I don't know")
  2. concept evidence   : curated concepts (data/rubric_overrides.json) or question-derived keywords
  3. misconception check: curated misconception patterns + generic dismissal patterns
  4. four dimensions    : correctness, relevance, completeness, explanation  (each 0-100)
  5. validated weighted aggregation (scoring_config.TECH_WEIGHTS)

IMPORTANT LIMITS (also shown in the UI): without a reference answer the system cannot verify facts. Correctness is
then an evidence-based proxy and is capped (LOW_EVIDENCE_CAP). Only questions with curated concepts get fuller grading.
"""
import json, math, re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import scoring_config as C
from paths import DATA_DIR

OVERRIDES_FILE = DATA_DIR / 'rubric_overrides.json'
EXTRA_STOP = {'explain', 'describe', 'tell', 'give', 'example', 'examples', 'difference', 'differences', 'between', 'does', 'did',
              'work', 'works', 'use', 'used', 'using', 'way', 'ways', 'common', 'main', 'different', 'various', 'one', 'would',
              'could', 'should', 'know', 'think', 'thing', 'things', 'kind', 'mean', 'means', 'really', 'also', 'make', 'makes'}
STOP = set(ENGLISH_STOP_WORDS) | EXTRA_STOP
FUNCTION_WORDS = set(ENGLISH_STOP_WORDS)

REASONING_CUES = ['because', 'therefore', 'so that', 'which means', 'as a result', 'for example', 'for instance', 'e.g', 'such as',
                  'whereas', 'while', 'however', 'in contrast', 'due to', 'leads to', 'results in', 'in order to', 'this means',
                  'hence', 'thus', 'unlike', 'instead of', 'compared to', 'on the other hand', 'since', 'when', 'including',
                  'by using', 'can use', 'to prevent', 'to avoid', 'so it', 'used to', 'used for']
DONT_KNOW = re.compile(r"\b(i\s*(do\s*not|don'?t|dont)\s*know|no\s*idea|not\s*sure|can'?t\s*(remember|recall)|cannot\s*(remember|recall)|skip)\b")
DISMISSAL = re.compile(r"\b(does\s*not\s*work\s*(at\s*all)?|doesn'?t\s*work(\s*at\s*all)?|is\s*useless|are\s*useless|no\s*use|not\s*(used|useful)\s*(at\s*all|anywhere)|a\s*random\s*thing|nothing\s*to\s*do\s*with)\b")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokens(text):
    return TOKEN_RE.findall(str(text or '').lower())


def stem(w):
    w = w.replace("'", '')
    for suf in ('ization', 'ational', 'ations', 'ation', 'ings', 'ing', 'ies', 'ied', 'ed', 'es', 'ly', 's'):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)] + ('y' if suf in ('ies', 'ied') else '')
    return w


def content_stems(text):
    return [stem(t) for t in tokens(text) if t not in STOP and len(t) > 2 and not t.isdigit()]


def _phrase_hit(concept, stems_seq, stem_set):
    """concept may be 'a|b' alternatives; each alternative is a word or short phrase."""
    for alt in concept.split('|'):
        parts = [stem(t) for t in tokens(alt)]
        if not parts:
            continue
        if len(parts) == 1:
            if parts[0] in stem_set:
                return True
        else:
            if not all(p in stem_set for p in parts):
                continue
            idx = [i for i, s in enumerate(stems_seq) if s == parts[0]]       # all parts within a short window
            for i in idx:
                window = stems_seq[max(0, i - len(parts) - 2): i + len(parts) + 3]
                if all(p in window for p in parts):
                    return True
    return False


def _topic_hit(q_stem, a_stems):
    if q_stem in a_stems:
        return True
    return len(q_stem) >= 6 and any(len(a) >= 5 and a[:5] == q_stem[:5] for a in a_stems)


def question_type(q):
    t = q.lower().strip()
    if re.match(r'^(what\s+(is|are|does)|define|what\'?s|name|list)\b', t) and len(t.split()) <= 12:
        return 'definition'
    if re.search(r'\b(design|architect|scal|trade-?off|implement|optimi[sz]e|how would you)\b', t):
        return 'design'
    return 'explain'


def estimate_difficulty(q):
    """Heuristic only (the source repos carry no difficulty labels): wording + length signals."""
    t = q.lower().strip(); n = len(t.split())
    adv = len(re.findall(r'\b(design|trade-?off|scal\w*|optimi[sz]\w*|compare|differen\w*|why|how would|implement\w*|architect\w*|distributed|concurren\w*)\b', t))
    beg_start = bool(re.match(r'^(what\s+(is|are)|define|name|list|what\'?s)\b', t))
    score = adv + (1 if n >= 14 else 0) - (2 if beg_start and n <= 10 else 0)
    if score <= -1:
        return 'Beginner'
    return 'Advanced' if score >= 2 else 'Intermediate'


def clamp(x, lo=0.0, hi=100.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    if math.isnan(x) or math.isinf(x):
        return lo
    return max(lo, min(hi, x))


# ---------------------------------------------------------------- validation / aggregation
def validate_dimensions(raw, expected_keys):
    """Validate evaluator output (e.g. from an external LLM, if one is ever plugged in).
    Returns {key: 0..100 float} or None if the structure is unusable."""
    if not isinstance(raw, dict):
        return None
    out = {}
    for k in expected_keys:
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return None
        out[k] = clamp(v)
    return out


def aggregate(dims, weights):
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f'rubric weights must sum to 1.0, got {total_w}')
    return clamp(sum(weights[k] * dims[k] for k in weights))


def fallback_evaluation(reason='The answer could not be evaluated automatically.', hr=False):
    keys = C.HR_WEIGHTS if hr else C.TECH_WEIGHTS
    return {'dimensions': {k: 0.0 for k in keys}, 'quality': 0.0, 'matched': [], 'missing': [], 'misconceptions': [],
            'flags': ['evaluator_error'], 'evidence_level': 'none', 'key_concepts': [], 'model_answer': None,
            'feedback': {'summary': reason, 'strengths': [], 'improvements': [reason]},
            'score': 0.0, 'correct': False, 'rubric': 'hr' if hr else 'technical'}


# ---------------------------------------------------------------- the scorer
class Scorer:
    def __init__(self, bank=None, overrides_path=OVERRIDES_FILE):
        self.overrides = []
        try:
            raw = json.loads(open(overrides_path, encoding='utf-8').read())
            for e in raw.get('entries', []):
                e['_re'] = re.compile(e['match'], re.I)
                e['_mis'] = [(re.compile(m['pattern'], re.I), m['message']) for m in e.get('misconceptions', [])]
                self.overrides.append(e)
        except (OSError, ValueError, KeyError, re.error):
            self.overrides = []
        self.domain_vocab = {}
        if bank is not None:
            self._build_domain_vocab(bank)

    def _build_domain_vocab(self, bank):
        """stems that recur across questions of the same role = on-topic vocabulary (supports RELEVANCE only)."""
        tech = bank[~bank.is_hr]
        for role, g in tech.groupby('role'):
            cnt = Counter()
            for q in g.question:
                cnt.update(set(content_stems(q)))
            self.domain_vocab[role] = {s for s, c in cnt.items() if c >= 2}

    def rubric_for(self, q):
        text = q['question']
        for e in self.overrides:
            if e['_re'].search(text):
                return e
        return None

    # -------- technical
    def evaluate_technical(self, q, answer, difficulty=None):
        try:
            return self._evaluate_technical(q, answer, difficulty)
        except Exception:     # never let one bad answer crash the interview
            return fallback_evaluation()

    def _evaluate_technical(self, q, answer, difficulty):
        question = str(q['question']); answer = str(answer or '').strip()
        role = q.get('role'); difficulty = difficulty or q.get('difficulty') or estimate_difficulty(question)
        flags = []
        toks = tokens(answer); n = len(toks)
        qtype = question_type(question)
        rub = self.rubric_for(q)
        q_stems = set(content_stems(question))
        a_seq = [stem(t) for t in toks]
        a_stem_set = set(a_seq)
        a_content = [s for s in content_stems(answer)]
        a_content_set = set(a_content)
        novel = a_content_set - q_stems

        # ---- concept model
        concepts, evidence = [], 'low'
        if rub:
            evidence = 'curated'
            for c in rub.get('core', []): concepts.append((c, 1.0, 'core'))
            for c in rub.get('important', []): concepts.append((c, 0.6, 'important'))
            for c in rub.get('optional', []): concepts.append((c, 0.3, 'optional'))
        else:
            kws = [k.strip().lower() for k in str(q.get('expected_keywords') or '').split(',') if k.strip() and k.strip() != 'nan']
            novel_kws = [k for k in kws if stem(k) not in q_stems]
            if novel_kws:
                evidence = 'keywords'
                concepts = [(k, 0.6, 'important') for k in novel_kws[:8]]
        matched, missing, wm, wt = [], [], 0.0, 0.0
        for c, w, kind in concepts:
            wt += w
            if _phrase_hit(c, a_seq, a_stem_set):
                matched.append(c.split('|')[0]); wm += w
            elif kind != 'optional':
                missing.append(c.split('|')[0])
        coverage = (wm / wt) if wt else None

        # ---- rule checks
        if n == 0:
            flags.append('empty')
        elif n < C.MIN_WORDS_FOR_CREDIT:
            flags.append('too_short')
        dont_know = bool(DONT_KNOW.search(answer.lower()))
        if dont_know: flags.append('dont_know')
        stop_ratio = (sum(t in FUNCTION_WORDS for t in toks) / n) if n else 0.0
        stuffing = n >= C.STUFFING_MIN_WORDS and stop_ratio < C.STUFFING_STOPWORD_RATIO
        if stuffing: flags.append('keyword_stuffing')
        echo = bool(a_content_set) and len(novel) < 2 and len(a_content_set & q_stems) >= 1
        if echo: flags.append('echoes_question')

        # ---- misconceptions
        misconceptions = []
        if rub:
            for pat, msg in rub['_mis']:
                if pat.search(answer):
                    misconceptions.append(msg)
        dismissal = bool(DISMISSAL.search(answer.lower()))
        if dismissal:
            misconceptions.append('The answer dismisses the topic ("does not work", "useless") instead of explaining it.')
        if misconceptions: flags.append('misconception')

        # ---- relevance (0-100)
        topic_cov = (sum(1 for qs in q_stems if _topic_hit(qs, a_content_set)) / len(q_stems)) if q_stems else 0.0
        domain = self.domain_vocab.get(role, set())
        domain_hits = len((a_content_set - q_stems) & domain)
        need = C.NEEDED_DOMAIN_TERMS.get(difficulty, 4)
        concept_support = (len(matched) / max(1, min(len(concepts), 3))) if concepts else None
        if concept_support is not None:
            support = min(1.0, concept_support)
        else:
            # generic on-topic vocabulary only counts if the answer also touches the question's own subject
            support = min(1.0, domain_hits / need) * min(1.0, topic_cov / 0.34)
        relevance = 100 * (0.35 * min(1.0, topic_cov / 0.5) + 0.65 * support)
        if len(novel) < 2:
            relevance = min(relevance, 25.0)       # repeating the question is not an answer
        if len(a_content_set) and topic_cov == 0 and support == 0:
            relevance = min(relevance, 8.0)

        # ---- completeness (0-100)
        if coverage is not None:
            completeness = 100 * coverage
        else:
            completeness = min(C.LOW_EVIDENCE_CAP, 100 * min(1.0, len(novel & (domain | a_content_set)) / (need * 2.0)) * (1.0 if domain_hits else 0.6))
            completeness = min(completeness, 100 * min(1.0, domain_hits / need) if domain else completeness)

        # ---- explanation (0-100)
        mn, tgt = C.LENGTH_EXPECTATION[qtype]
        length_score = clamp((n - 4) / max(1.0, tgt - 4), 0, 1)
        low = ' '.join(toks)
        cues = sum(1 for cue in REASONING_CUES if cue in low)
        cue_score = 1.0 if qtype == 'definition' else min(1.0, cues / (2.0 if qtype == 'design' else 1.0))
        prose = clamp((stop_ratio - 0.10) / 0.15, 0, 1)
        explanation = 100 * (0.4 * length_score + 0.35 * cue_score + 0.25 * prose)
        if qtype == 'definition' and n >= mn and len(matched) >= max(1, len(concepts) // 2 if concepts else 1):
            explanation = max(explanation, 80.0)       # a concise, on-point definition is not penalised for brevity

        # ---- correctness (0-100): evidence-based proxy
        if coverage is not None:
            correctness = 100 * coverage
            if evidence == 'keywords':
                correctness = min(correctness, 85.0)
        else:
            correctness = min(C.LOW_EVIDENCE_CAP, min(relevance, completeness) * 0.9 + 0.1 * explanation * 0.5)
            flags.append('low_reference_evidence')
        correctness *= min(1.0, 0.5 + 0.5 * (relevance / 100.0)) if relevance < 50 else 1.0
        if misconceptions:
            correctness *= C.MISCONCEPTION_FACTOR
            if dismissal and not any(m for m in misconceptions[:-1]): correctness = min(correctness, 100 * C.DISMISSAL_FACTOR * (coverage or 0.3))
        if stuffing:
            correctness *= C.STUFFING_CORRECTNESS_FACTOR; explanation *= C.STUFFING_EXPLANATION_FACTOR
        if n < C.MIN_WORDS_FOR_CREDIT or dont_know:
            correctness, relevance, completeness, explanation = [min(v, 10.0) for v in (correctness, relevance, completeness, explanation)]
        if n == 0:
            correctness = relevance = completeness = explanation = 0.0

        dims = validate_dimensions({'correctness': correctness, 'relevance': relevance, 'completeness': completeness,
                                    'explanation': explanation}, C.TECH_WEIGHTS)
        if dims is None:
            return fallback_evaluation()
        quality = aggregate(dims, C.TECH_WEIGHTS)
        key_concepts = [c.split('|')[0] for c, w, k in concepts if k != 'optional']
        fb = self._feedback_tech(dims, quality, matched, missing, misconceptions, flags, n, qtype, evidence, cues, rub)
        return {'dimensions': dims, 'quality': quality, 'matched': matched, 'missing': missing, 'misconceptions': misconceptions,
                'flags': flags, 'evidence_level': evidence, 'key_concepts': key_concepts,
                'model_answer': (rub or {}).get('model_answer'), 'feedback': fb, 'score': quality / 100.0,
                'correct': quality >= C.PASS_THRESHOLD, 'rubric': 'technical', 'question_type': qtype, 'difficulty': difficulty}

    def _feedback_tech(self, d, quality, matched, missing, mis, flags, n, qtype, evidence, cues, rub):
        s, imp = [], []
        if 'empty' in flags:
            return {'summary': 'No answer was submitted.', 'strengths': [], 'improvements': ['Submit an answer to receive a score.']}
        if 'dont_know' in flags:
            imp.append('You said you did not know. Even a partial attempt (a definition, one example) earns credit.')
        if matched:
            s.append('You covered: ' + ', '.join(matched[:5]) + '.')
        if missing:
            imp.append('You did not mention: ' + ', '.join(missing[:5]) + '.')
        for m in mis:
            imp.append('Possible factual problem: ' + m)
        if d['relevance'] >= 70: s.append('Your answer addressed the question that was asked.')
        elif d['relevance'] < 35: imp.append('Your answer stayed off the topic of the question or only repeated its wording.')
        if 'echoes_question' in flags: imp.append('You mostly repeated the words of the question; add your own explanation.')
        if 'keyword_stuffing' in flags: imp.append('The answer reads like a list of keywords. Write full sentences that explain how the ideas connect.')
        if 'too_short' in flags: imp.append('The answer was too short to assess.')
        elif d['explanation'] >= 70: s.append('The explanation was clear' + (' and included reasoning.' if cues else '.'))
        elif d['explanation'] < 45 and 'keyword_stuffing' not in flags:
            imp.append('Explain the reasoning behind your answer' + (' and give an example.' if qtype != 'definition' else '.')
                       + f' ({n} words; a fuller answer to this type of question is usually longer).')
        if 'low_reference_evidence' in flags:
            imp.append('Note: this question has no reference answer in the bank, so correctness and completeness are capped estimates.')
        if quality >= 80 and not imp: summary = 'Strong answer.'
        elif quality >= C.PASS_THRESHOLD: summary = 'Acceptable answer with gaps.'
        elif quality >= 35: summary = 'Partially correct or incomplete answer.'
        else: summary = 'Weak answer: little relevant or correct content was found.'
        return {'summary': summary, 'strengths': s, 'improvements': imp}

    # -------- HR / behavioural
    def evaluate_hr(self, q, answer):
        try:
            return self._evaluate_hr(q, answer)
        except Exception:
            return fallback_evaluation(hr=True)

    def _evaluate_hr(self, q, answer):
        question = str(q['question']); answer = str(answer or '').strip()
        toks = tokens(answer); n = len(toks); low = answer.lower()
        flags = []
        q_stems = set(content_stems(question)); a_content = set(content_stems(answer)); novel = a_content - q_stems
        topic_cov = (len(a_content & q_stems) / len(q_stems)) if q_stems else 0.0
        relevance = 100 * (0.5 * min(1.0, topic_cov / 0.35) + 0.5 * min(1.0, len(novel) / 8.0))
        if len(novel) < 3: relevance = min(relevance, 25.0)
        digits = len(re.findall(r'\b\d+(?:[.,]\d+)?%?', answer))
        proper = len(re.findall(r'(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z]{2,}', answer))
        specificity = 100 * min(1.0, 0.45 * min(1.0, n / 70.0) + 0.25 * min(1.0, digits / 1.0) + 0.30 * min(1.0, proper / 2.0))
        narrative = len(re.findall(r"\b(when|last\s+(?:year|semester|month|summer)|during|in\s+my|at\s+my|our\s+(?:project|team)|my\s+(?:team|project|college|internship)|i\s+was|i\s+had|i\s+worked|there\s+was|one\s+time|once)\b", low))
        past = len(re.findall(r'\bi\s+(?:\w+\s+)?\w+ed\b', low)) + len(re.findall(r'\b(?:led|built|made|took|ran|wrote|got|found|began|chose|decided|spoke|taught|met)\b', low))
        hypothetical = len(re.findall(r"\b(i\s+would|i\s+always|i\s+usually|i\s+generally|i\s+believe\s+in|i\s+will)\b", low))
        example = 100 * min(1.0, 0.6 * min(1.0, narrative / 2.0) + 0.4 * min(1.0, past / 2.0))
        if hypothetical >= 1 and narrative == 0: example = min(example, 25.0); flags.append('no_specific_example')
        elif narrative == 0 and past == 0: flags.append('no_specific_example')
        action_verbs = len(re.findall(r"\bi\s+(?:\w+\s+)?(?:led|built|decided|organi[sz]ed|talked|asked|created|fixed|implemented|resolved|proposed|planned|coordinated|wrote|developed|presented|suggested|divided|worked|spoke|discussed|reviewed|learned|taught|tested|designed|managed|volunteered|started)\b", low))
        action = 100 * min(1.0, action_verbs / 3.0)
        result_hits = len(re.findall(r"\b(result|outcome|as\s+a\s+result|ended\s+up|achieved|improved|reduced|increased|learned|learnt|lesson|delivered|successfully|saved|completed|finished|won|resolved|helped)\b|\d+\s*%", low))
        outcome = 100 * min(1.0, result_hits / 2.0)
        if n < C.MIN_WORDS_FOR_CREDIT:
            relevance, specificity, example, action, outcome = [min(v, 10.0) for v in (relevance, specificity, example, action, outcome)]
            flags.append('too_short')
        if n == 0:
            relevance = specificity = example = action = outcome = 0.0; flags.append('empty')
        dims = validate_dimensions({'relevance': relevance, 'specificity': specificity, 'example': example, 'action': action, 'outcome': outcome}, C.HR_WEIGHTS)
        if dims is None:
            return fallback_evaluation(hr=True)
        quality = aggregate(dims, C.HR_WEIGHTS)
        star = {'situation': example >= 50, 'task': bool(re.search(r"\b(my\s+role|responsible|goal|had\s+to|needed\s+to|task|assigned)\b", low)),
                'action': action >= 50, 'result': outcome >= 50}
        s, imp = [], []
        if 'empty' in flags: imp.append('No answer was submitted.')
        else:
            if dims['relevance'] >= 65: s.append('Your answer stayed on the question.')
            else: imp.append('Your answer did not clearly address what the question asked.')
            if dims['example'] >= 60: s.append('You described a concrete situation.')
            else: imp.append('Describe one specific situation (where, when, what was at stake) instead of general statements.')
            if dims['action'] >= 60: s.append('You explained what you personally did.')
            else: imp.append('Say what YOU did, using "I" statements and specific steps.')
            if dims['outcome'] >= 60: s.append('You stated an outcome or lesson.')
            else: imp.append('End with the result or what you learned, with a number if you can.')
            if dims['specificity'] < 40 and 'too_short' not in flags: imp.append(f'Add specifics (names, numbers, tools); the answer was {n} words.')
        summary = 'Strong, specific answer.' if quality >= 75 else 'Reasonable answer, needs more specifics.' if quality >= C.PASS_THRESHOLD else 'Answer lacks a specific example or outcome.' if quality >= 25 else 'Very limited answer.'
        return {'dimensions': dims, 'quality': quality, 'matched': [], 'missing': [], 'misconceptions': [], 'flags': flags,
                'evidence_level': 'heuristic', 'key_concepts': [], 'model_answer': None, 'star': star,
                'feedback': {'summary': summary, 'strengths': s, 'improvements': imp}, 'score': quality / 100.0,
                'correct': None, 'rubric': 'hr'}
