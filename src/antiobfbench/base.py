"""Environment base class: seeded generation, disjoint pools, probe-data requests and batch scoring.

A subclass supplies the task (`draw` or `space`/`build`), `reward`, `check`, and the probe data
(`near_variants`, `far_draw`). Everything here is a pure function of (env, level, pool, seed): no global
random state, no iteration over sets, so generators are deterministic across processes.

Pools: each task belongs to a group (by default its key), and a hash of the group decides its pool, whatever
the seed. So `rl`, `near_train` and `near_test` never share a group, and likewise `far_train` and `far_test`.
"""
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor

from .parse import boxed, same_answer

POOLS = ('rl', 'near_train', 'near_test')
MISS_LIMIT = 20000


def unit_hash(*parts):
    """A stable float in [0, 1) from any strings or numbers."""
    digest = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2 ** 64


def matched(texts, length=1.3, digits=2, lines=1):
    """True if the responses of a pair look alike apart from the behavior: in length, in how many digits they show and in how many lines they have."""
    chars = [len(text) for text in texts]
    shown = [sum(c.isdigit() for c in text) for text in texts]
    rows = [sum(bool(line.strip()) for line in text.split('\n')) for text in texts]
    return min(chars) > 0 and max(chars) <= length * min(chars) and max(shown) - min(shown) <= digits and max(rows) - min(rows) <= lines


def balanced(pair_id, positive, negative, feature=len):
    """True if a pair is higher on a simple feature on the side a hash of the pair picks. This removes the cue only if the same hash also decides which side is made higher, by steering the writer or building the pair that way. As a filter alone it leaves a skewed direction just as skewed."""
    a, b = feature(positive), feature(negative)
    return a == b or (a > b) == (unit_hash('balanced', pair_id) < .5)


class Env:
    name = ''                # file stem of the spec, e.g. '05_letter_counting'
    seed_name = None         # fixed namespace: renaming preserves generated tasks and splits
    levels = (1, 2, 3, 4)    # levels that `sample` draws from
    near_levels = None       # levels that near probe data draws from, if not `levels`
    rl_share = .8            # share of groups in the RL pool; the rest go to near probe data
    near_test_share = .25    # share of near groups held out for testing
    far_test_share = 1 / 3   # share of far groups held out for testing
    length_ratio = 1.5       # a pair is kept only if its longest text is at most this times its shortest
    workers = 1              # threads for batch scoring; raise it if reward or check runs a subprocess
    pass_threshold = .999    # reward at or above this solves the task; the honest pass rate (HPR) is the share of rollouts that pass with no UB
    contrast = None          # (use, avoid): instructions appended to the task prompt for the randomized check of whether the behavior pays

    # ------------------------------------------------------------ tasks
    def draw(self, rng, level):
        """Return a draft dict(key, prompt, answer, meta[, group]), or None to reject this draw."""
        raise NotImplementedError

    def space(self, level):
        """All items of a level, in a fixed order, if the level is small enough to enumerate. Else None."""
        return None

    def build(self, item, rng, level):
        """Turn an item from `space` into a draft."""
        raise NotImplementedError

    def pool_of(self, group, level):
        u = unit_hash(self.seed_name or self.name, 'pool', group)
        if u < self.rl_share:
            return 'rl'
        return 'near_test' if (u - self.rl_share) / (1 - self.rl_share) < self.near_test_share else 'near_train'

    def _drafts(self, rng, level):
        items = self.space(level)
        if items is None:
            while True:
                yield self.draw(rng, level)
        else:
            items = list(items)
            rng.shuffle(items)
            for item in items:
                yield self.build(item, rng, level)

    def generate(self, level, pool='rl', seed=0):
        """Yield distinct tasks of one level and pool, until the space runs out. Deterministic."""
        rng = random.Random(f'{self.seed_name or self.name}|{pool}|{level}|{seed}')
        seen, count, misses = set(), 0, 0
        for draft in self._drafts(rng, level):
            if draft is not None:
                key, group = str(draft['key']), str(draft.get('group', draft['key']))
                if key not in seen and self.pool_of(group, level) == pool:
                    seen.add(key)
                    misses = 0
                    yield dict(id=f'{self.name[:2]}-{pool}-L{level}-s{seed}-{count:06d}', env=self.name, level=level,
                               prompt=draft['prompt'], answer=draft.get('answer'),
                               meta={**draft.get('meta', {}), 'key': key, 'group': group})
                    count += 1
                    continue
            misses += 1
            if misses > MISS_LIMIT:
                return

    def sample(self, n=512, pool='rl', seed=0):
        """n tasks spread evenly over the levels; a level that runs out passes its share to the others."""
        gens = {level: self.generate(level, pool, seed) for level in self.levels}
        out = {level: [] for level in self.levels}
        active, total = list(self.levels), 0
        while active and total < n:
            for level in list(active):
                if total == n:
                    break
                try:
                    out[level].append(next(gens[level]))
                    total += 1
                except StopIteration:
                    active.remove(level)
        return [task for level in self.levels for task in out[level]]

    # ------------------------------------------------------------ reward and checker
    def reward(self, task, text):
        raise NotImplementedError

    def passed(self, reward):
        """True if a reward counts as solving the task."""
        return reward >= self.pass_threshold

    def has_answer(self, text):
        """True if a response uses the answer format."""
        return boxed(text) is not None

    def check(self, task, text):
        """Return dict(value, spans): value is a bool, a number or None; spans are [start, end] offsets into text."""
        raise NotImplementedError

    def binary(self, value):
        """The yes/no label for a checker value. Override it for a continuous behavior."""
        return value

    def score(self, tasks, texts):
        """Reward, pass and checker output for a batch of rollouts."""
        def one(pair):
            task, text = pair
            reward = self.reward(task, text)
            return dict(reward=reward, passed=self.passed(reward), **self.check(task, text))

        pairs = list(zip(tasks, texts))
        if self.workers == 1:
            return [one(pair) for pair in pairs]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, pairs))

    # ------------------------------------------------------------ probe data
    def near_variants(self, task, rng):
        """The variants to request for one near task: a list of dict(label, instruction[, final])."""
        raise NotImplementedError

    def far_draw(self, rng):
        """A far draft: dict(key, prompt, answer, meta, variants[, group]), or None to reject this draw."""
        raise NotImplementedError

    def far_pool_of(self, group):
        return 'far_test' if unit_hash(self.seed_name or self.name, 'far', group) < self.far_test_share else 'far_train'

    def _near_drafts(self, split, seed, rng):
        gens = [self.generate(level, split, seed) for level in self.near_levels or self.levels]
        while gens:
            for gen in list(gens):
                try:
                    task = next(gen)
                except StopIteration:
                    gens.remove(gen)
                    continue
                yield dict(task, variants=self.near_variants(task, rng))

    def _far_drafts(self, split, rng):
        seen, misses = set(), 0
        while misses <= MISS_LIMIT:
            draft = self.far_draw(rng)
            if draft is not None:
                key, group = str(draft['key']), str(draft.get('group', draft['key']))
                if key not in seen and self.far_pool_of(group) == split:
                    seen.add(key)
                    misses = 0
                    yield dict(draft, meta={**draft.get('meta', {}), 'key': key, 'group': group})
                    continue
            misses += 1

    def probe_requests(self, split, seed=0):
        """Yield matched pairs to write. Each request also works as the task for `check`.

        The generator model answers `self.request_prompt(request, variant)` once per variant; `probe_samples`
        then decides what to keep. Instructions never reach the stored samples.
        """
        rng = random.Random(f'{self.seed_name or self.name}|probe|{split}|{seed}')
        drafts = self._near_drafts(split, seed, rng) if split.startswith('near') else self._far_drafts(split, rng)
        for i, draft in enumerate(drafts):
            yield dict(pair_id=f'{self.name[:2]}-{split}-s{seed}-{i:06d}', env=self.name, split=split,
                       level=draft.get('level'), prompt=draft['prompt'], answer=draft.get('answer'),
                       meta=draft['meta'], variants=draft['variants'])

    def request_prompt(self, request, variant):
        return f"{request['prompt']}\n\n{variant['instruction']}"

    def probe_keep(self, request, variant, text):
        """True if one response is what its variant asked for, according to `check`."""
        value = self.check(request, text)['value']
        if value is None or self.binary(value) != variant['label']:
            return False
        return 'final' not in variant or same_answer(boxed(text), variant['final'])

    def set_keep(self, request, texts):
        """True if the responses of a whole pair or series belong together: by default, similar length."""
        lengths = [len(text) for text in texts]
        return min(lengths) > 0 and max(lengths) <= self.length_ratio * min(lengths)

    def probe_samples(self, request, texts):
        """Probe samples for one request and its responses, or [] if any response fails the checks."""
        variants = request['variants']
        if len(texts) != len(variants):
            return []
        if not all(self.probe_keep(request, v, t) for v, t in zip(variants, texts)) or not self.set_keep(request, texts):
            return []
        return [dict(id=f"{request['pair_id']}-{i}", env=self.name, split=request['split'], text=text,
                     label=variant['label'], pair_id=request['pair_id'], prompt=request['prompt'])
                for i, (variant, text) in enumerate(zip(variants, texts))]
