"""Focused checks for optional, isolated tracking; no external requests or real credentials."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import wandb_logger as tracking
from rl_plots import PLOTS, TrainingPlots


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'logs')
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_disabled_has_no_side_effects(self):
        with patch.object(tracking.subprocess, 'Popen') as launch:
            self.assertIsNone(tracking.start_logger(self.directory, {}))
            tracking.finish_logger(self.directory, {}, 0)
        launch.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_start_failure_never_escapes_or_logs_error_contents(self):
        with patch.object(tracking.subprocess, 'Popen', side_effect=OSError('sensitive details')):
            self.assertIsNone(tracking.start_logger(self.directory, {'wandb': {'enabled': True}}))
        saved = (self.directory / 'wandb/status.json').read_text()
        self.assertNotIn('sensitive details', saved)
        self.assertEqual(json.loads(saved)['error_type'], 'OSError')

    def test_worker_gets_no_key_in_command_or_inherited_environment(self):
        with patch.dict(os.environ, {'WANDB_API_KEY': 'fake-test-key'}):
            with patch.object(tracking.subprocess, 'Popen') as launch:
                tracking.start_logger(self.directory, {'wandb': {'enabled': True}})
        args, kwargs = launch.call_args
        self.assertNotIn('fake-test-key', repr(args))
        self.assertNotIn('WANDB_API_KEY', kwargs['env'])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(kwargs['stdout'], tracking.subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], tracking.subprocess.DEVNULL)

    def test_all_plot_redraws_survive_overwrites_and_failures_are_optional(self):
        plots = TrainingPlots(self.directory, keep_history=True)
        plots.update({'step': 1, **{k: .1 for k in PLOTS}})
        image = self.directory / 'wandb/plot_queue/000001/mean_reward.png'
        original = image.read_bytes()
        plots.update({'step': 2, **{k: .8 for k in PLOTS}})
        self.assertEqual(image.read_bytes(), original)
        self.assertNotEqual((self.directory / 'plots/mean_reward.png').read_bytes(), original)
        self.assertTrue((self.directory / 'wandb/plot_queue/000002/complete.json').exists())
        with patch('rl_plots.os.link', side_effect=OSError('no space')):
            plots.update({'step': 3, **{k: .5 for k in PLOTS}})
        self.assertTrue((self.directory / 'plots/mean_reward.png').exists())

    def test_same_metrics_config_and_delayed_images_reach_sdk(self):
        config = {'wandb': {'enabled': True, 'project': 'antiobfbench'}}
        (self.directory / 'resolved_config.json').write_text(json.dumps(config))
        rows = [{'step': 1, 'loss': -.25, 'probe_accuracy': None, 'by_source': {'math': {'n': 8}}},
                {'step': 2, 'loss': .12, 'probe_accuracy': .71}]
        (self.directory / 'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        plots = self.directory / 'wandb/plot_queue/000001'
        plots.mkdir(parents=True)
        (plots / 'mean_reward.png').write_bytes(b'image fixture')
        (plots / 'complete.json').write_text('{"step":1}')
        tracking.finish_logger(self.directory, config, 0)
        run = MagicMock()
        run.id, run.url = 'test-id', 'https://example.invalid/test'
        sdk = SimpleNamespace(init=MagicMock(return_value=run), Settings=lambda **kw: kw,
                              util=SimpleNamespace(generate_id=lambda: 'test-id'), Image=lambda path: path)
        with patch.dict(sys.modules, {'wandb': sdk}), \
             patch.dict(os.environ, {'ANTIOBF_MACHINE': 'olivia', 'ANTIOBF_WANDB_KEY_FILE': '/tmp/fake-credential'}), \
             patch.object(tracking.os, 'nice'), patch.object(tracking, 'read_key', return_value='fake-test-key') as read:
            tracking.upload(self.directory, os.getpid())
        read.assert_called_once_with(Path('/tmp/fake-credential'))
        self.assertEqual(sdk.init.call_args.kwargs['tags'], ['olivia'])
        self.assertEqual(sdk.init.call_args.kwargs['config'], config)
        self.assertEqual(sdk.init.call_args.kwargs['name'], self.directory.name)
        self.assertEqual([c.args[0] for c in run.log.call_args_list[:2]], rows)
        self.assertEqual(run.log.call_args_list[2].args[0]['step'], 1)
        self.assertIn('plots/mean_reward', run.log.call_args_list[2].args[0])
        status = json.loads((self.directory / 'wandb/status.json').read_text())
        self.assertEqual(status['last_metric_step'], 2)
        self.assertEqual(status['plot_steps'], [1])
        run.finish.assert_called_once_with(exit_code=0)


if __name__ == '__main__':
    unittest.main()
