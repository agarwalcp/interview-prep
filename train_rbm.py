"""
Step 2: trains the RBM (Units II-IV: latent variable, Gibbs sampling, Contrastive Divergence)
on SYNTHETIC behaviour profiles and saves weights + 'strong'/'weak' anchors.
Run once:  python train_rbm.py
Visible units: [answer_score, fast_response, motion_calm, vocal_steadiness]
"""
import numpy as np
from rbm import RBM, make_synthetic_behavior_data
from paths import DATA_DIR, RBM_WEIGHTS, RBM_ANCHORS

def main():
    np.random.seed(0)
    data, labels = make_synthetic_behavior_data(n=600, seed=1)
    rbm = RBM(n_visible=4, n_hidden=2, lr=0.05, seed=7)
    rbm.train(data.copy(), epochs=300, k=1, batch_size=32, verbose=False)
    h = rbm.hidden_representation(data)
    strong_h, weak_h = h[labels == 1].mean(axis=0), h[labels == 0].mean(axis=0)
    DATA_DIR.mkdir(exist_ok=True)
    rbm.save(str(RBM_WEIGHTS))
    np.savez(str(RBM_ANCHORS), strong_h=strong_h, weak_h=weak_h)
    print('strong anchor:', strong_h.round(3), ' weak anchor:', weak_h.round(3))
    print('separation   :', float(np.abs(strong_h - weak_h).sum().round(3)), '(should be large, ~1.6)')
    print('saved', RBM_WEIGHTS.name, 'and', RBM_ANCHORS.name, 'in', DATA_DIR)

if __name__ == '__main__':
    main()
