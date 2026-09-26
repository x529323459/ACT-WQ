"""Lightweight production defaults, also used by independent experiment runners."""


def configure(args, granularity=None):
    # Explicit experimental arms may override granularity, never bit widths.
    if granularity is None:
        granularity = getattr(args, '_experiment_granularity', None)
    if granularity is None and getattr(args, '_force_channel_experiment', False):
        granularity = 'channel'
    if granularity not in (None, 'channel', 'tensor', 'per-channel', 'per-tensor'):
        raise ValueError('Unknown experimental granularity: ' + str(granularity))
    args.gptq_weight_perchannel = (args.gptq_bits != 8 if granularity is None
                                  else granularity in ('channel', 'per-channel'))
    args.gptq_activation_perchannel = (args.gptq_act_bits != 8 if granularity is None
                                      else granularity in ('channel', 'per-channel'))
    args.gptq_weight_range_search = args.gptq_bits == 8
    args.gptq_input_centering = getattr(args, 'gptq_input_centering', False)
    return args
