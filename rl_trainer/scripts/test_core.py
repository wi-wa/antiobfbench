"""Small numerical regression tests; no model downloads or GPU required."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'src')]
import torch
import torch.nn.functional as F
from overlap import generate_overlapped
from rl_core import advantages, collate_trajectories, optimization_batches, validate_resume_config
from rl_loss import backward_with_retry, frozen_head_loss
from antiobfbench.rl_env import RLEnvironment, eligible_groups
from config import load_config
from runtime import gpu_environment, output_directory


class CoreTests(unittest.TestCase):
    def test_scheduler_gpu_assignment_and_output_directory(self):
        config = {'run': {'gpu': 7}}
        env = {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': 'GPU-allocated'}
        self.assertEqual(gpu_environment(config, env), env)
        self.assertNotIn('CUDA_VISIBLE_DEVICES', gpu_environment(config, {'SLURM_JOB_ID': '123'}))
        self.assertEqual(gpu_environment(config, {})['CUDA_VISIBLE_DEVICES'], '7')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(output_directory(ROOT, 'example'), ROOT / 'logs/example')
        with patch.dict(os.environ, {'ANTIOBF_RUNS_DIR': '/tmp/antiobf-test-runs'}):
            self.assertEqual(output_directory(ROOT, 'example'), Path('/tmp/antiobf-test-runs/example'))

    def test_default_task_paths_for_all_environments(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'logs') as directory:
            path = Path(directory) / 'config.json'
            for number in range(1, 10):
                with self.subTest(environment=number):
                    path.write_text(json.dumps({'environment': f'{number:02}'}))
                    config = load_config(path)
                    rows = RLEnvironment(config).load(config['data']['path'])
                    self.assertGreaterEqual(len(rows), config['rl']['prompts_per_rollout'])

    def test_machine_preset_sits_between_defaults_and_run_file(self):
        defaults = json.loads((ROOT / 'config.json').read_text())
        preset = json.loads((ROOT / 'olivia_fast.json').read_text())
        self.assertEqual(load_config(machine='elsewhere')['inference']['kv_tokens'], defaults['inference']['kv_tokens'])
        self.assertEqual(load_config(machine='olivia')['inference']['kv_tokens'], preset['inference']['kv_tokens'])
        with patch.dict(os.environ, {'ANTIOBF_MACHINE': 'olivia'}), tempfile.TemporaryDirectory(dir=ROOT / 'logs') as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'inference': {'kv_tokens': 65536}}))
            config = load_config(path)
            self.assertEqual(config['inference']['kv_tokens'], 65536)
            self.assertEqual(config['inference']['mem_fraction'], preset['inference']['mem_fraction'])

    def test_leave_one_out_and_constant_groups(self):
        actual = advantages([-1, 0, 1, 1, 1, 1], 3, 'leave_one_out')
        torch.testing.assert_close(actual, torch.tensor([-1.5, 0., 1.5, 0., 0., 0.]))
        self.assertEqual(eligible_groups([-1, 0, 1, 1, 1, 1], 3), [True]*3 + [False]*3)

    def test_prompt_padding_and_terminal_token_mask(self):
        rows = [dict(prompt_ids=[1, 2], output_ids=[3, 4]),
                dict(prompt_ids=[5], output_ids=[6])]
        ids, attention, mask = collate_trajectories(rows, 0, 'cpu')
        self.assertEqual(ids.tolist(), [[1, 2, 3, 4], [5, 6, 0, 0]])
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(ids[:, 1:][mask].tolist(), [3, 4, 6])

    def test_signed_loss_and_gradients_match_direct_autograd(self):
        torch.manual_seed(9)
        weight = torch.randn(17, 7)
        targets = torch.tensor([1, 2, 3, 4, 5])
        # Full rollout count is 8, including trajectories with zero advantages.
        coefficients = torch.tensor([1.5, -1.5, 0., .25, -.25]) / 8
        for temperature, softcap in [(1., None), (.7, 2.)]:
            with self.subTest(temperature=temperature, softcap=softcap):
                hidden = torch.randn(5, 7, requires_grad=True)
                reference = hidden.detach().clone().requires_grad_(True)
                logits = F.linear(reference, weight)
                if softcap:
                    logits = torch.tanh(logits / softcap) * softcap
                expected = (F.cross_entropy(logits / temperature, targets, reduction='none') * coefficients).sum()
                actual = frozen_head_loss(hidden, weight, targets, coefficients,
                                          temperature, softcap, chunk_size=2)
                expected.backward()
                actual.backward()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(hidden.grad, reference.grad)
                self.assertEqual(hidden.grad[2].abs().sum().item(), 0)

    def test_microbatch_packing_preserves_all_trajectories(self):
        rows = [dict(prompt_ids=[1], output_ids=[2]*n) for n in (1, 2, 3, 20)]
        groups = list(optimization_batches(rows, range(4), 3, max_tokens=6))
        self.assertEqual(groups, [[0, 1], [2], [3]])

    def test_oom_retry_discards_partial_gradients_without_stepping(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=.1)
        attempts = []
        def backward(size):
            attempts.append(size)
            (parameter * (100 if size > 1 else 3)).backward()
            if size > 1:
                raise torch.OutOfMemoryError('simulated')
            return 3.
        result, size = backward_with_retry(backward, optimizer, 4)
        self.assertEqual((result, size, attempts), (3., 1, [4, 2, 1]))
        self.assertEqual(parameter.grad.item(), 3)
        self.assertEqual(parameter.item(), 1)
        self.assertFalse(optimizer.state)

    def test_probe_sign_format_and_group_gates(self):
        for name, direction in [('03_farthest_north', -1), ('09_math_dinosaurs', 1)]:
            env = RLEnvironment({'environment': name, 'data': {}})
            self.assertAlmostEqual(env.combine(1, .8, True, .5, True), 1 + direction*.4)
            self.assertEqual(env.combine(1, .8, True, .5, False), 1)
            self.assertEqual(env.combine(-1, .8, False, .5, True), -1)

    def test_overlapped_generation_scores_during_generation(self):
        import asyncio
        import threading

        class Engine:
            loop = asyncio.new_event_loop()

            async def async_generate(self, input_ids, sampling_params, lora_path, return_logprob, stream):
                assert stream and len(input_ids) == len(sampling_params) == len(lora_path)
                assert all(options['stream_interval'] > 4096 for options in sampling_params)

                async def results():
                    for index in range(len(input_ids)):          # one partial message per request first
                        yield dict(index=index, output_ids=[input_ids[index][0]], meta_info=dict(finish_reason=None))
                    for index in sorted(range(len(input_ids)), key=lambda i: sampling_params[i]['delay']):
                        await asyncio.sleep(.005)
                        yield dict(index=index, output_ids=[input_ids[index][0], sampling_params[index]['delay']],
                                   adapter=lora_path[index], meta_info=dict(finish_reason=dict(type='stop')))
                return results()

        delays = [.03, .01, .02, .04, .05, .01]
        seen, grouped, main = [], [], threading.get_ident()

        def on_response(index, response):
            seen.append((index, threading.get_ident() != main))
            return index * 10

        def on_groups(groups, responses):
            grouped.append([(g, all(responses[i] is not None for i in range(2 * g, 2 * g + 2))) for g in groups])

        responses, results = generate_overlapped(Engine(), [[i] for i in range(6)], [dict(delay=d) for d in delays],
                                                 'policy-1', 2, on_response, on_groups, workers=3)
        self.assertEqual([r['output_ids'][0] for r in responses], list(range(6)))
        self.assertEqual(results, [0, 10, 20, 30, 40, 50])
        self.assertEqual(sorted(i for i, threaded in seen if threaded), list(range(6)))
        self.assertEqual(sorted(g for batch in grouped for g, complete in batch if complete), [0, 1, 2])
        with self.assertRaisesRegex(RuntimeError, 'scoring failed'):
            def broken(groups, responses):
                raise RuntimeError('scoring failed')
            generate_overlapped(Engine(), [[i] for i in range(2)], [dict(delay=0)] * 2, 'policy-1', 2, None, broken)
        with self.assertRaisesRegex(KeyError, 'check failed'):
            def failing(index, response):
                raise KeyError('check failed')
            generate_overlapped(Engine(), [[i] for i in range(2)], [dict(delay=0)] * 2, 'policy-1', 2, failing, None)

    def test_resume_rejects_objective_changes(self):
        saved = json.loads((ROOT / 'config.json').read_text())
        current = copy.deepcopy(saved)
        current['optim']['microbatch_size'] = 1
        current['optim']['share_prompts'] = not saved['optim']['share_prompts']
        validate_resume_config(saved, current)
        current['optim']['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'optim'):
            validate_resume_config(saved, current)


if __name__ == '__main__':
    unittest.main()
