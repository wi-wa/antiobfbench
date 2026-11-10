"""Generate a rollout batch and start scoring responses while later ones are still being generated."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

QUIET_STREAM = 1 << 20     # tokens between partial outputs of a streaming request


def generate_overlapped(engine, prompts, sampling, adapter, group_size, on_response=None, on_groups=None, workers=1):
    """Like `engine.generate` on a list of requests, with two optional callbacks that run during generation.

    `on_response(index, response)` runs on a pool of `workers` threads as each response arrives.
    `on_groups(groups, responses)` runs on one more thread each time every request of one or more prompt groups
    (`group_size` consecutive requests) has finished. Returns the responses and the `on_response` results, both in
    request order. An error in a callback is raised once generation has ended.
    """
    count = len(prompts)
    if count % group_size:
        raise ValueError('Expected complete prompt groups')
    responses, futures = [None] * count, [None] * count
    remaining = [group_size] * (count // group_size)
    ready, errors = queue.Queue(), []

    def score_ready_groups():
        finished = False
        while not finished:
            groups = [ready.get()]
            while True:
                try:
                    groups.append(ready.get_nowait())
                except queue.Empty:
                    break
            finished = None in groups
            groups = [g for g in groups if g is not None]
            if groups and not errors:
                try:
                    on_groups(groups, responses)
                except BaseException as error:      # reported by the caller's thread below
                    errors.append(error)

    pool = ThreadPoolExecutor(max_workers=workers) if on_response else None
    scorer = threading.Thread(target=score_ready_groups, daemon=True) if on_groups else None
    if scorer:
        scorer.start()

    async def everything():
        # One batch request, so that all prompts reach the scheduler together, in streaming mode so that each
        # response is delivered when it finishes. The long stream interval keeps partial output to one early
        # message per request.
        stream = await engine.async_generate(
            input_ids=prompts, sampling_params=[dict(options, stream_interval=QUIET_STREAM) for options in sampling],
            lora_path=[adapter] * count, return_logprob=False, stream=True)
        async for result in stream:
            index = result['index']
            if result['meta_info'].get('finish_reason') is None or responses[index] is not None:
                continue
            responses[index] = result
            if pool:
                futures[index] = pool.submit(on_response, index, result)
            group = index // group_size
            remaining[group] -= 1
            if remaining[group] == 0 and scorer:
                ready.put(group)

    try:
        engine.loop.run_until_complete(everything())
    finally:
        if scorer:
            ready.put(None)
            scorer.join()
        if pool:
            pool.shutdown(wait=True)
    if errors:
        raise errors[0]
    if any(response is None for response in responses):
        raise RuntimeError('Incomplete rollout batch')
    return responses, [future.result() for future in futures] if pool else None
