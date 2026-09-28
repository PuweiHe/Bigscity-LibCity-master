import unittest

import torch

from scripts.reproduce_seq2seq import load_model


class Seq2SeqTests(unittest.TestCase):
    def build(self, rnn="GRU", **kwargs):
        return load_model()(
            {
                "input_window": 4,
                "output_window": 2,
                "hidden_size": 8,
                "rnn_type": rnn,
                "decoder_start": "last",
                **kwargs,
            },
            {"num_nodes": 2, "feature_dim": 2, "output_dim": 1},
        )

    def test_determinism_target_independence_and_dtype(self):
        for rnn in ["GRU", "lstm", "RNN"]:
            model = self.build(rnn).double().eval()
            x = torch.randn(3, 4, 2, 2, dtype=torch.float64)
            with torch.no_grad():
                a = model.predict({"X": x})
                b = model.predict({"X": x, "y": torch.full((3, 2, 2, 1), float("nan"))})
            self.assertEqual(a.shape, (3, 2, 2, 1))
            self.assertEqual(a.dtype, x.dtype)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(a).all())

    def test_training_gradients_and_teacher_forcing(self):
        for rnn in ["GRU", "LSTM", "RNN"]:
            model = self.build(rnn, teacher_forcing_ratio=1).train()
            batch = {"X": torch.randn(3, 4, 2, 2), "y": torch.randn(3, 2, 2, 1)}
            model(batch).square().mean().backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
            )
            with self.assertRaises(KeyError):
                model({"X": batch["X"]})

    def test_legacy_random_and_validation(self):
        model = self.build(decoder_start="random").eval()
        batch = {"X": torch.randn(3, 4, 2, 2)}
        with torch.no_grad():
            self.assertFalse(torch.equal(model(batch), model(batch)))
        with self.assertRaises(ValueError):
            self.build(decoder_start="invalid")

    def test_tiny_batch_overfit(self):
        torch.manual_seed(7)
        model = self.build().train()
        x = torch.randn(4, 4, 2, 2)
        y = x[:, -1:, :, :1].repeat(1, 2, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = (model({"X": x}) - y).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.05)
