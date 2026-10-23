import numpy as np
from rbm import RBM
from paths import RBM_WEIGHTS, RBM_ANCHORS

class ConfidenceModel:
    """Turns RBM hidden activations into a 0-1 'confidence/preparedness' score by projecting
    onto the line between the strong-profile and weak-profile hidden anchors."""
    def __init__(self, weights=RBM_WEIGHTS, anchors=RBM_ANCHORS):
        import os
        if not (os.path.exists(weights) and os.path.exists(anchors)):
            raise FileNotFoundError('RBM weights not found. Run  python train_rbm.py  once (see README step 4).')
        self.rbm = RBM.load(weights)
        a = np.load(anchors)
        self.strong, self.weak = a['strong_h'], a['weak_h']
        # used for sensors that are switched off: the training-data mean (a neutral value)
        self.defaults = self.rbm.mean_.copy()

    def score(self, features):
        f = np.atleast_2d(np.array(features, dtype=float))
        h = self.rbm.hidden_representation(f)
        d = self.strong - self.weak
        return np.clip(((h - self.weak) @ d) / float(d @ d), 0, 1)
