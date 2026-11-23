# Consistency loss

For binary predictions p = sigmoid(z) and q = sigmoid(z_view), condition C uses

    L = 0.5 * (BCEWithLogits(z, y) + BCEWithLogits(z_view, y))
        + consistency_weight * JS(Bernoulli(p), Bernoulli(q))

JS uses natural logarithms. Its per-image range is [0, ln(2)]. Both views
receive gradients; neither branch is detached. Log probabilities and their
mixture are computed in log space. FP16/BF16 inputs are promoted to FP32.

```python
from src.losses import paired_view_loss

# z and z_view: raw AI logits, shape (N,) or (N, 1).
# y: labels, 0 = real, 1 = AI, same batch order for both views.
loss = paired_view_loss(z, z_view, y, consistency_weight=1.0)
loss.backward()
```

Use `consistency_weight=0.0` for condition B, and `1.0` for C as specified in
the proposal. The default reduction is the batch mean; `none` returns shape
(N,), and `sum` sums the per-image losses. Condition A uses clean-view BCE.

This module accepts a single binary logit per image. If the model returns two
class logits ordered [real, AI], use `z = output[:, 1] - output[:, 0]`.
Do not pass probabilities. The training entry point and loader integration
still need to be agreed with TONG and OUYANG.

Run the checks from the repository root with PyTorch installed:

```bash
python -m unittest discover -s tests -p "test_consistency.py" -v
```

The tests use synthetic inputs and require no dataset or pretrained weights.
They check the formula against a probability-space reference, bounds,
symmetry, gradients, B/C equivalence, extreme logits and CUDA mixed precision
when CUDA is available. They do not constitute a training pilot.
