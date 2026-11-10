"""Memory-bounded REINFORCE loss for a frozen language-model output head."""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def backward_with_retry(backward_fn, optimizer, microbatch_size):
    """Retry gradient accumulation only; never step the optimizer inside backward_fn."""
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            return backward_fn(microbatch_size), microbatch_size
        except torch.OutOfMemoryError:
            if microbatch_size == 1:
                raise
            microbatch_size = max(1, microbatch_size // 2)
            print(f'CUDA OOM: retrying the same rollout batch with optimization microbatch {microbatch_size}.',
                  flush=True)
        # Leave the exception scope first so its traceback releases activation tensors.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


class _FrozenHeadLoss(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, hidden, weight, targets, coefficients, temperature, softcap, chunk_size, backend):
        if weight.requires_grad or coefficients.requires_grad:
            raise ValueError('Output-head weights and reward coefficients must be frozen.')
        gradient = torch.empty_like(hidden)
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        # Compute dLoss/dHidden while each chunk's logits are live. Retain only
        # this small gradient instead of checkpointing/recomputing the head GEMM.
        for start in range(0, hidden.shape[0], chunk_size):
            end = start + chunk_size
            if backend == 'liger_fused' and temperature == 1.0 and softcap is None:
                from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
                # Supply signed token weights before the gradient projection.
                # The frozen head needs only dHidden, never a vocabulary-sized dWeight.
                losses, _, _, _, chunk_gradient, _, _ = fused_linear_cross_entropy_forward(
                    hidden[start:end], weight, targets[start:end], reduction='none',
                    token_grad_output=coefficients[start:end], compute_gradients=True,
                    weight_requires_grad=False,
                    chunk_mem_const=(weight.shape[0] + hidden.shape[1] - 1) // hidden.shape[1])
                gradient[start:end] = chunk_gradient
                total.add_((losses * coefficients[start:end]).sum())
                continue
            with torch.enable_grad():
                chunk = hidden[start:end].detach().requires_grad_(True)
                logits = F.linear(chunk, weight)
                if softcap is not None:
                    logits = torch.tanh(logits / softcap) * softcap
                logits = logits.float() / temperature
                if backend in ('liger', 'liger_fused'):
                    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
                    losses = LigerCrossEntropyFunction.apply(
                        logits, targets[start:end], None, -100, 0.0, 0.0, 'none')[0]
                else:
                    losses = F.cross_entropy(logits, targets[start:end], reduction='none')
                loss = (losses * coefficients[start:end]).sum()
                gradient[start:end] = torch.autograd.grad(loss, chunk)[0]
            total.add_(loss.detach())
        ctx.save_for_backward(gradient)
        return total

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    @once_differentiable
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        return gradient * grad_output, None, None, None, None, None, None, None


def frozen_head_loss(hidden, weight, targets, coefficients, temperature=1.0, softcap=None,
                     chunk_size=512, backend='torch'):
    """Weighted sum of token cross-entropies, with first-order hidden gradients."""
    if backend not in ('torch', 'liger', 'liger_fused'):
        raise ValueError(f'Unknown frozen-head loss backend: {backend}')
    return _FrozenHeadLoss.apply(hidden, weight, targets, coefficients, temperature, softcap, chunk_size, backend)
