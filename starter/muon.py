"""Muon optimizer for 2D hidden-layer weight matrices.

Source idea: Keller Jordan's nanoGPT-speedrun project — fastest possible
loss convergence within a small, fixed step budget, which is exactly this
assignment's regime. Pure PyTorch (matmuls only), no external package.

What it does: takes the momentum-accumulated gradient of a 2D weight
matrix and replaces it with its nearest semi-orthogonal matrix via a few
iterations of the quintic Newton-Schulz iteration (no SVD needed).
Intuition: an orthogonalized update spreads the step evenly across all
singular directions instead of letting a few dominant directions absorb
most of it — reported to converge faster per-step than plain Adam
momentum on transformer hidden-layer matrices specifically.

Only use on 2D matrices. Embeddings/norms/biases should go to Adam/AdamW.
"""
import torch


@torch.no_grad()
def _zeropower_via_newton_schulz(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                         ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                assert g.ndim == 2, "Muon is only for 2D weight matrices"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                u = _zeropower_via_newton_schulz(update, steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u, alpha=-lr * scale)