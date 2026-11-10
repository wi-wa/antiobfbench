"""SGLang entrypoint with a local fix for Gemma's linear RoPE scaling."""

from sglang.srt.models.gemma3_causal import Gemma3TextModel


_original_init = Gemma3TextModel.__init__


def _init_with_linear_rope(self, config, *args, **kwargs):
    _original_init(self, config, *args, **kwargs)
    params = config.rope_parameters
    global_params = params.get('full_attention', params)
    # This installed SGLang builds the native global rotary embedding as default
    # RoPE even for linear scaling. Linear scaling divides frequencies by factor.
    if (global_params.get('rope_type') == 'linear'
            and self.rotary_emb.rope_type == 'default'):
        factor = float(global_params['factor'])
        if factor <= 0:
            raise ValueError('RoPE factor must be positive')
        self.rotary_emb.inv_freq.div_(factor)
        self.rotary_emb.original_inv_freq = self.rotary_emb.inv_freq


# Spawned SGLang workers import this entrypoint, so the fix applies there too.
Gemma3TextModel.__init__ = _init_with_linear_rope
