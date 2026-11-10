"""Headless, per-run training curves with trailing eight-step smoothing."""

from collections import deque
from io import BytesIO
import json
import os
from pathlib import Path


PLOTS = {
    'mean_reward': ('Mean reward', '#13836d', (-1, 1)),
    'mean_accuracy': ('Correct and valid fraction', '#2762a8', (0, 1)),
    'mean_correct_format': ('Strict-format fraction', '#9c3a85', (0, 1)),
    'mean_thinklen': ('Reasoning tokens', '#b77e16', None),
}


def moving_average(values, window=8):
    if type(window) is not int or window < 1:
        raise ValueError('The moving-average window must be a positive integer.')
    recent, result, total = deque(), [], 0.0
    for value in values:
        recent.append(value)
        total += value
        if len(recent) > window:
            total -= recent.popleft()
        result.append(total / len(recent))
    return result


class TrainingPlots:
    def __init__(self, run_directory, keep_history=False):
        # Lazy imports keep plotting dependencies out of spawned inference workers.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        self.directory = Path(run_directory) / 'plots'
        self.keep_history = keep_history
        self.directory.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.values = {name: [] for name in PLOTS}
        self.figure = Figure(figsize=(8, 4.5), dpi=120)
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots()
        self.axes.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        self.axes.set_xlabel('Training step')
        self.axes.grid(True, color='#e2e5e9', linewidth=0.7)
        self.axes.set_axisbelow(True)
        self.axes.spines[['top', 'right']].set_visible(False)
        self.line, = self.axes.plot([], [], linewidth=2, marker='o', markersize=3)
        self.figure.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.87)

    def update(self, metrics):
        step = metrics['step']
        if self.steps and step <= self.steps[-1]:
            raise ValueError('Plot steps must increase.')
        values = {name: float(metrics[name]) for name in PLOTS}
        self.steps.append(step)
        for name, (label, color, limits) in PLOTS.items():
            self.values[name].append(values[name])
            smoothed = moving_average(self.values[name])
            self.line.set_data(self.steps, smoothed)
            self.line.set_color(color)
            self.axes.set_title(f'{name} (8-step moving average)', pad=14)
            self.axes.set_ylabel(label)
            self.axes.set_xlim(self.steps[0] - 0.5, self.steps[-1] + 0.5)
            self.axes.set_ylim(*(limits or (0, max(1.0, max(smoothed) * 1.1))))
            buffer = BytesIO()
            self.figure.savefig(buffer, format='png')
            path = self.directory / f'{name}.png'
            # Readers see either the previous complete PNG or the new one.
            temporary = path.with_suffix('.png.tmp')
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(path)
        if self.keep_history:
            # Hard links retain each redraw for the separate uploader without copying PNG data.
            # Tracking failures must not affect training or the ordinary local plots.
            try:
                history = self.directory.parent / 'wandb' / 'plot_queue' / f'{step:06d}'
                history.mkdir(parents=True, exist_ok=True)
                for name in PLOTS:
                    destination = history / f'{name}.png'
                    if not destination.exists():
                        os.link(self.directory / destination.name, destination)
                (history / 'complete.json').write_text(json.dumps({'step': step}) + '\n')
            except OSError:
                pass
