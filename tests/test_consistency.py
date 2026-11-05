import math
import unittest

import torch
from torch.nn import functional as F

from src.losses import bernoulli_js_divergence, paired_view_loss


class ConsistencyLossTests(unittest.TestCase):
    def test_matches_probability_reference(self):
        p = torch.tensor([0.1, 0.3, 0.8], dtype=torch.float64)
        q = torch.tensor([0.9, 0.4, 0.2], dtype=torch.float64)
        p_dist = torch.stack((1 - p, p), dim=-1)
        q_dist = torch.stack((1 - q, q), dim=-1)
        m = (p_dist + q_dist) / 2
        expected = 0.5 * (
            (p_dist * torch.log(p_dist / m)).sum(-1)
            + (q_dist * torch.log(q_dist / m)).sum(-1)
        )
        actual = bernoulli_js_divergence(torch.logit(p), torch.logit(q), "none")
        torch.testing.assert_close(actual, expected)

    def test_identical_views_have_zero_js(self):
        x = torch.tensor([-10000.0, -2.0, 0.0, 2.0, 10000.0])
        torch.testing.assert_close(bernoulli_js_divergence(x, x, "none"), torch.zeros_like(x))

    def test_symmetry_and_bounds(self):
        x = torch.linspace(-30, 30, 101, dtype=torch.float64)
        q = -x + 0.7
        values = bernoulli_js_divergence(x, q, "none")
        torch.testing.assert_close(values, bernoulli_js_divergence(q, x, "none"))
        self.assertTrue(torch.all(values >= 0))
        self.assertTrue(torch.all(values <= math.log(2) + 1e-12))

    def test_opposite_confident_predictions_reach_log_two(self):
        value = bernoulli_js_divergence(torch.tensor([-10000.0]), torch.tensor([10000.0]))
        self.assertAlmostEqual(value.item(), math.log(2), places=6)

    def test_weight_zero_is_condition_b_including_gradients(self):
        x = torch.tensor([-1.5, 0.2, 2.0], requires_grad=True)
        q = torch.tensor([0.8, -0.7, 1.0], requires_grad=True)
        y = torch.tensor([0.0, 1.0, 1.0])
        expected = 0.5 * (F.binary_cross_entropy_with_logits(x, y)
                          + F.binary_cross_entropy_with_logits(q, y))
        actual = paired_view_loss(x, q, y, consistency_weight=0)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, (x, q), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (x, q))
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)

    def test_condition_c_adds_weighted_js(self):
        x, q, y = torch.tensor([-1.0, 0.2]), torch.tensor([2.0, -0.5]), torch.tensor([0, 1])
        b = paired_view_loss(x, q, y, consistency_weight=0, reduction="none")
        expected = b + 2.5 * bernoulli_js_divergence(x, q, "none")
        torch.testing.assert_close(paired_view_loss(x, q, y, 2.5, "none"), expected)

    def test_js_backpropagates_through_both_views(self):
        x = torch.tensor([-1.0, 0.3], requires_grad=True)
        q = torch.tensor([0.5, -0.8], requires_grad=True)
        bernoulli_js_divergence(x, q).backward()
        for branch in (x, q):
            self.assertTrue(torch.all(torch.isfinite(branch.grad)))
            self.assertTrue(torch.all(branch.grad.abs() > 0))

    def test_finite_difference_gradients(self):
        x = torch.tensor([-1.2, 0.3], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.0, 1.0], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(bernoulli_js_divergence, (x, q)))
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: paired_view_loss(a, b, y), (x, q)))

    def test_reductions_and_column_vectors(self):
        x, q, y = torch.tensor([-2.0, 1.0]), torch.tensor([0.0, -1.0]), torch.tensor([0, 1])
        for loss in (lambda a, b, r: bernoulli_js_divergence(a, b, r),
                     lambda a, b, r: paired_view_loss(a, b, y[:, None], reduction=r)):
            values = loss(x[:, None], q[:, None], "none")
            self.assertEqual(values.shape, (2,))
            torch.testing.assert_close(loss(x, q, "mean"), values.mean())
            torch.testing.assert_close(loss(x, q, "sum"), values.sum())

    def _check_extremes(self, device, dtype):
        x = torch.tensor([-10000, 10000, 0, -20], device=device, dtype=dtype, requires_grad=True)
        q = torch.tensor([10000, -10000, 0, 20], device=device, dtype=dtype, requires_grad=True)
        y = torch.tensor([1, 0, 1, 0], device=device)
        value = paired_view_loss(x, q, y)
        self.assertTrue(torch.isfinite(value))
        if dtype in (torch.float16, torch.bfloat16):
            self.assertEqual(value.dtype, torch.float32)
        value.backward()
        for branch in (x, q):
            self.assertTrue(torch.all(torch.isfinite(branch.grad)))

    def test_cpu_extreme_logits(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_extremes("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision(self):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self._check_extremes("cuda", torch.float16)
            self._check_extremes("cuda", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_agree(self):
        x, q, y = torch.tensor([-3.0, 1.0, 0.1]), torch.tensor([1.0, -0.5, 0.2]), torch.tensor([0, 1, 0])
        cpu = paired_view_loss(x, q, y)
        gpu = paired_view_loss(x.cuda(), q.cuda(), y.cuda())
        torch.testing.assert_close(cpu, gpu.cpu())

    def test_rejects_invalid_inputs(self):
        x, q, y = torch.zeros(2), torch.ones(2), torch.tensor([0, 1])
        bad_calls = [
            lambda: bernoulli_js_divergence(x, torch.zeros(3)),
            lambda: bernoulli_js_divergence(torch.zeros(2, 2), q),
            lambda: bernoulli_js_divergence(torch.zeros(0), torch.zeros(0)),
            lambda: bernoulli_js_divergence(x, q, "invalid"),
            lambda: paired_view_loss(x, q, torch.tensor([0, 2])),
            lambda: paired_view_loss(x, q, torch.zeros(3)),
        ]
        for weight in (-1, float("nan"), float("inf")):
            bad_calls.append(lambda weight=weight: paired_view_loss(x, q, y, weight))
        for call in bad_calls:
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            bernoulli_js_divergence(y, y)


if __name__ == "__main__":
    unittest.main()
