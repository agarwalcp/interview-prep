from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / 'data'          # always <project>/data, whatever the working directory
QUESTION_BANK = DATA_DIR / 'question_bank.csv'
RBM_WEIGHTS = DATA_DIR / 'rbm_weights.npz'
RBM_ANCHORS = DATA_DIR / 'rbm_anchors.npz'
