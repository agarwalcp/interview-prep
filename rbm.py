"""
Restricted Boltzmann Machine, implemented from scratch (Units III-IV).

Visible units (4, binary/probabilistic in [0,1]):
    [correct, fast_response, low_motion, vocal_steadiness]
Hidden units (2): learned latent factors -- expected to specialise into
something like "confidence" and "preparedness" after training, though the
model discovers this itself; we don't hand-label which hidden unit means what.

Trained via k-step Contrastive Divergence (CD-k), computing the gradient of
the log-likelihood approximately, exactly as described in Units III-IV of
the syllabus.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(n_visible, n_hidden))
        self.b_visible = np.zeros(n_visible)
        self.b_hidden = np.zeros(n_hidden)
        self.lr = lr
        self.mean_ = np.zeros(n_visible)
        self.std_ = np.ones(n_visible)

    def standardize(self, v, fit=False):
        if fit:
            self.mean_ = v.mean(axis=0)
            self.std_ = v.std(axis=0) + 1e-6
        return (v - self.mean_) / self.std_

    def sample_hidden(self, v):
        p_h = sigmoid(v @ self.W + self.b_hidden)
        h = (np.random.rand(*p_h.shape) < p_h).astype(float)
        return p_h, h

    def sample_visible(self, h):
        # Gaussian-Bernoulli RBM: visible units are real-valued (mean-field,
        # no re-binarization) -- the right choice for continuous behavioral
        # features in [0,1], and standard practice to reduce gradient noise.
        mean_v = h @ self.W.T + self.b_visible
        return mean_v, mean_v

    def contrastive_divergence_step(self, v0, k=1):
        """One weight update using CD-k (Gibbs sampling for k steps)."""
        p_h0, h0 = self.sample_hidden(v0)
        v_k, h_k = v0, h0
        p_hk = p_h0
        for _ in range(k):
            _, v_k = self.sample_visible(h_k)
            p_hk, h_k = self.sample_hidden(v_k)

        # gradient of log-likelihood, approximated via CD-k
        positive_grad = v0.T @ p_h0
        negative_grad = v_k.T @ p_hk

        n = v0.shape[0]
        self.W += self.lr * (positive_grad - negative_grad) / n
        self.b_visible += self.lr * np.mean(v0 - v_k, axis=0)
        self.b_hidden += self.lr * np.mean(p_h0 - p_hk, axis=0)

        recon_error = np.mean((v0 - v_k) ** 2)
        return recon_error

    def train(self, data, epochs=200, k=1, batch_size=16, verbose=True):
        data = self.standardize(data, fit=True)
        n = data.shape[0]
        for epoch in range(epochs):
            np.random.shuffle(data)
            errs = []
            for start in range(0, n, batch_size):
                batch = data[start:start + batch_size]
                errs.append(self.contrastive_divergence_step(batch, k=k))
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  reconstruction_error={np.mean(errs):.4f}")

    def hidden_representation(self, v):
        """Latent 'confidence/preparedness' representation for a session."""
        v = self.standardize(v, fit=False)
        p_h, _ = self.sample_hidden(v)
        return p_h

    def save(self, path):
        np.savez(path, W=self.W, b_visible=self.b_visible, b_hidden=self.b_hidden,
                  mean_=self.mean_, std_=self.std_)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        rbm = cls(d["W"].shape[0], d["W"].shape[1])
        rbm.W, rbm.b_visible, rbm.b_hidden = d["W"], d["b_visible"], d["b_hidden"]
        rbm.mean_, rbm.std_ = d["mean_"], d["std_"]
        return rbm


def make_synthetic_behavior_data(n=400, seed=0):
    """
    Synthetic stand-in for real logged interview sessions (Phase 1 dataset
    swap-in target: MIT Interview Dataset / ChaLearn First Impressions v2).
    Two latent clusters: "strong candidates" and "weak candidates", each
    generating correlated [correct, fast, low_motion, steadiness] features.
    """
    rng = np.random.default_rng(seed)
    n_strong = n // 2
    n_weak = n - n_strong

    strong = rng.normal(loc=[0.85, 0.8, 0.75, 0.8], scale=0.12, size=(n_strong, 4))
    weak = rng.normal(loc=[0.3, 0.35, 0.3, 0.35], scale=0.15, size=(n_weak, 4))

    data = np.clip(np.vstack([strong, weak]), 0, 1)
    labels = np.array([1] * n_strong + [0] * n_weak)
    idx = rng.permutation(n)
    return data[idx], labels[idx]


if __name__ == "__main__":
    data, labels = make_synthetic_behavior_data(n=600, seed=1)
    rbm = RBM(n_visible=4, n_hidden=2, lr=0.05, seed=7)
    rbm.train(data.copy(), epochs=300, k=1, batch_size=32)

    hidden = rbm.hidden_representation(data)
    # sanity check: does the learned hidden layer separate strong vs weak?
    strong_h = hidden[labels == 1].mean(axis=0)
    weak_h = hidden[labels == 0].mean(axis=0)
    print("\nMean hidden activation, strong candidates:", strong_h)
    print("Mean hidden activation, weak candidates:  ", weak_h)
    rbm.save("data/rbm_weights.npz")
    print("\nSaved trained weights to data/rbm_weights.npz")
