"""
All tunable numbers for answer scoring and delivery analysis live here.
These are INITIAL rubric settings, not scientifically validated values. Tune them with a labelled
validation set (see tests/README_labelling.md) before making any accuracy claim.
"""

SCORING_VERSION = '2.0-rubric'

# ---- technical answers: weights must sum to 1.0, every dimension is scored 0-100 first ----
TECH_WEIGHTS = {'correctness': 0.40, 'relevance': 0.25, 'completeness': 0.20, 'explanation': 0.15}

# ---- HR / behavioural answers (separate rubric) ----
HR_WEIGHTS = {'relevance': 0.20, 'specificity': 0.20, 'example': 0.20, 'action': 0.20, 'outcome': 0.20}

PASS_THRESHOLD = 60.0          # quality >= this counts as "correct" for the Bayesian network / Markov chain

# When a question has no reference concepts (only the question's own words), correctness / completeness
# cannot be verified. They are capped so such answers can never look like verified top marks.
LOW_EVIDENCE_CAP = 65.0
NEEDED_DOMAIN_TERMS = {'Beginner': 3, 'Intermediate': 4, 'Advanced': 6}   # distinct on-topic terms for full proxy credit

MISCONCEPTION_FACTOR = 0.35    # correctness is multiplied by this for every matched misconception (min 1 match)
DISMISSAL_FACTOR = 0.30        # "it does not work", "useless", ...
STUFFING_STOPWORD_RATIO = 0.12 # answers with fewer function words than this look like keyword lists
STUFFING_MIN_WORDS = 6
STUFFING_CORRECTNESS_FACTOR = 0.4
STUFFING_EXPLANATION_FACTOR = 0.3
MIN_WORDS_FOR_CREDIT = 3       # fewer words than this: every dimension <= 10

# expected answer length (words) per question type: (minimum, target)
LENGTH_EXPECTATION = {'definition': (10, 25), 'explain': (25, 60), 'design': (40, 90)}

# ---- delivery (spoken answers only) ----
FILLER_FREE_RATE = 1.0         # % of words; up to this rate there is no penalty (a single "um" is harmless)
FILLER_ZERO_RATE = 10.0        # % of words at which the filler component reaches 0
IDEAL_WPM = (110.0, 170.0)     # no pace penalty inside this range
PACE_ZERO_WPM = (60.0, 230.0)  # pace component reaches 0 at these speeds
REPEAT_ZERO_RATE = 0.10        # repeated-phrase ratio at which the repetition component reaches 0
DELIVERY_WEIGHTS = {'fillers': 0.50, 'pace': 0.25, 'repetition': 0.25}   # renormalised over the available components
MIN_WORDS_FOR_DELIVERY = 15    # shorter transcripts give no delivery score (too little evidence)
MIN_SECONDS_FOR_PACE = 5.0
ALWAYS_FILLERS = ['um', 'umm', 'uh', 'uhh', 'erm', 'er', 'hmm', 'hmmm']
HABITUAL_FILLERS = ['basically']       # counted only when used repeatedly (>= HABITUAL_MIN times in one answer)
HABITUAL_MIN = 2
