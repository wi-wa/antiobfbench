"""Prompt sharing must give the same probe features, loss and gradients as running every row in full.

Uses a small randomly initialised Gemma 3 text model on the CPU, with both sliding-window and full-attention
layers and sequences longer than the window. No model download or GPU is needed.
"""
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'src')]
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from antiobfbench.mean_probe import pooled_features, pooled_features_shared
from antiobfbench.prefix_sharing import length_batches
from reinforce_worker import backward_rollouts, backward_rollouts_shared

PAD = 0
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def small_model(seed=0):
    torch.manual_seed(seed)
    config = Gemma3TextConfig(vocab_size=96, hidden_size=32, intermediate_size=48, num_hidden_layers=7,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
                              max_position_embeddings=256, pad_token_id=PAD, eos_token_id=1, bos_token_id=2,
                              attn_implementation='sdpa')
    model = Gemma3ForCausalLM(config).eval()
    assert {'sliding_attention', 'full_attention'} <= set(config.layer_types)
    return model


def rollouts(seed, groups=4, members=5):
    rng = random.Random(seed)
    rows = []
    for _ in range(groups):
        prompt = [rng.randrange(3, 96) for _ in range(rng.choice([3, 6, 11, 14]))]
        for _ in range(members):
            # Lengths from 1 token to well past the sliding window of 8.
            rows.append(dict(prompt_ids=prompt, output_ids=[rng.randrange(3, 96) for _ in range(rng.choice([1, 2, 5, 9, 17, 30]))]))
    return rows


class PromptSharingTests(unittest.TestCase):
    def test_batches_keep_every_row_once(self):
        costs = [5, 1, 9, 2, 2, 30]
        order = sorted(range(6), key=costs.__getitem__)
        batches = list(length_batches(costs, order, 3, 12))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(6)))
        self.assertTrue(all(len(b) <= 3 and (len(b) == 1 or max(costs[i] for i in b) * len(b) <= 12) for b in batches))

    def test_probe_features_match_the_full_forward_at_every_layer(self):
        model = small_model()
        rows = rollouts(1)
        for layer in range(1, model.config.num_hidden_layers + 1):
            with self.subTest(layer=layer):
                shared = pooled_features_shared(model, rows, layer, PAD, token_budget=64, max_rows=7)
                full = pooled_features(model, rows, [layer], PAD, batch_size=4, token_budget=10 ** 6)[:, 0]
                np.testing.assert_allclose(shared, full, rtol=2e-4, atol=2e-5)

    def test_loss_and_gradients_match_the_full_forward(self):
        for seed, attention in [(2, 'sdpa'), (3, 'eager')]:
            with self.subTest(attention=attention):
                model = small_model(seed)
                model.config._attn_implementation = attention
                policy = get_peft_model(model, LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0, target_modules=TARGETS,
                                                          bias='none', task_type='CAUSAL_LM'))
                torch.manual_seed(seed)
                for name, parameter in policy.named_parameters():
                    if 'lora_B' in name:
                        torch.nn.init.normal_(parameter, std=.3)
                jobs = rollouts(seed + 10)
                coefficients = torch.randn(len(jobs))
                coefficients[::4] = 0                      # rows with zero advantage are left out, as in training
                active = sorted((i for i in range(len(jobs)) if coefficients[i] != 0),
                                key=lambda i: len(jobs[i]['prompt_ids']) + len(jobs[i]['output_ids']))
                parameters = [p for p in policy.parameters() if p.requires_grad]

                def gradients(function, **options):
                    policy.zero_grad(set_to_none=True)
                    loss = function(policy, jobs, active, coefficients, PAD, 1.0, 16, **options)
                    return loss, [p.grad.clone() for p in parameters]

                expected_loss, expected = gradients(backward_rollouts, microbatch_size=3, max_tokens=120)
                # No budget (one prompt pass), a small budget (several prompt passes), and a budget below one prompt.
                for options in [dict(microbatch_size=16, max_tokens=0), dict(microbatch_size=1, max_tokens=60),
                                dict(microbatch_size=2, max_tokens=10)]:
                    actual_loss, actual = gradients(backward_rollouts_shared, **options)
                    self.assertAlmostEqual(actual_loss, expected_loss, delta=1e-4 * max(1.0, abs(expected_loss)))
                    scale = max(g.abs().max().item() for g in expected)
                    self.assertGreater(scale, 1e-4)
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, rtol=2e-3, atol=2e-5 * scale)


if __name__ == '__main__':
    unittest.main()
