# -*- coding: utf-8 -*-

"""Network related utility tools."""

import logging
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


import torch


def check_nan(x, name="Tensor"):
    """
    Checks if a Tensor, a list of Tensors, or a dictionary of Tensors contains NaN values and prints related information.

    Parameters:
    x (torch.Tensor, list of torch.Tensor, or dict): The Tensor, list of Tensors, or dictionary of Tensors to check.
    name (str): The name of the Tensor for printing purposes. Defaults to 'Tensor'.

    Returns:
    bool: Returns True if any Tensor in the input contains NaN values, otherwise returns False.
    """

    def check_tensor(tensor, tensor_name):
        """
        Checks for NaN values in a single Tensor and prints detailed information.
        """
        nan_mask = torch.isnan(tensor)
        has_nan = torch.any(nan_mask)

        if has_nan:
            nan_count = torch.sum(nan_mask).item()
            if nan_count == tensor.numel():
                print(f"Tensor '{tensor_name}' contains only NaN values.")
            else:
                non_nan_tensor = tensor[~nan_mask]
                min_value = non_nan_tensor.min().item()
                max_value = non_nan_tensor.max().item()
                mean_value = non_nan_tensor.mean().item()

                print(f"Tensor '{tensor_name}' contains NaN values.")
                print(f"Shape: {tensor.shape}")
                print(f"Number of NaNs: {nan_count}")
                print(f"Min value (excluding NaNs): {min_value}")
                print(f"Max value (excluding NaNs): {max_value}")
                print(f"Mean value (excluding NaNs): {mean_value}")

        return has_nan

    if isinstance(x, dict):
        # If input is a dictionary, check each Tensor in the dictionary
        has_nan_in_any_tensor = False
        for key, tensor in x.items():
            if isinstance(tensor, torch.Tensor):
                if check_tensor(tensor, f"{name}, key {key}"):
                    has_nan_in_any_tensor = True
            else:
                print(f"Item '{key}' is not a Tensor. Skipping...")
        return has_nan_in_any_tensor

    elif isinstance(x, list):
        # If input is a list, check each Tensor in the list
        has_nan_in_any_tensor = False
        for i, tensor in enumerate(x):
            if isinstance(tensor, torch.Tensor):
                if check_tensor(tensor, f"{name}, index {i}"):
                    has_nan_in_any_tensor = True
            else:
                print(f"Item at index {i} is not a Tensor. Skipping...")
        return has_nan_in_any_tensor

    elif isinstance(x, torch.Tensor):
        # If input is a single Tensor
        return check_tensor(x, name)

    else:
        raise TypeError(
            "Input must be either a Tensor, a list of Tensors, or a dictionary of Tensors."
        )


def build_aligned_inputs_and_targets(
    input,
    input_length,
    start_token=0,
    stop_token=0,
    detach_input=False,
    detach_target=True,
):
    assert input.ndim in [2, 3], "input dimension error"
    inp = F.pad(input, (1, 0), value=start_token)
    if input.ndim == 3:
        input_list = unpad_sequence(
            input.transpose(1, 2), input_length.cpu(), batch_first=True
        )
        input_list = [F.pad(d, (0, 0, 0, 1), value=stop_token) for d in input_list]
        tar = pad_sequence(input_list, batch_first=True).transpose(1, 2)
    else:
        input_list = unpad_sequence(input, input_length.cpu(), batch_first=True)
        input_list = [F.pad(d, (0, 1), value=stop_token) for d in input_list]
        tar = pad_sequence(input_list, batch_first=True)
    input_length = input_length + 1
    assert input.size(-1) + 1 == inp.size(-1) == tar.size(-1), "input length error"
    return (
        inp.detach() if detach_input else inp,
        tar.detach() if detach_target else tar,
        input_length,
    )


def pad_unpad_sequence(
    text_embed, text_token_len, feats, feats_len, target, target_len, IGNORE_ID=-1
):
    text_embed = unpad_sequence(text_embed, text_token_len.cpu(), batch_first=True)
    feats = unpad_sequence(feats, feats_len.cpu(), batch_first=True)
    target = unpad_sequence(target, target_len.cpu(), batch_first=True)
    lm_input = [
        torch.concat([text_embed[i], feats[i], target[i]], dim=0)
        for i in range(len(text_embed))
    ]
    lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
    lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
    return lm_input, lm_input_len


def split_hidden_states(
    last_hidden_state, text_token_len, prompt_feats_len, target_feats_len, IGNORE_ID=-1
):
    batch_size = last_hidden_state.size(0)
    text_logits = unpad_sequence(
        last_hidden_state, text_token_len.cpu(), batch_first=True
    )
    text_logits = pad_sequence(text_logits, batch_first=True, padding_value=IGNORE_ID)

    prompt_logits = [
        last_hidden_state[i].narrow(0, text_token_len[i], prompt_feats_len[i])
        for i in range(batch_size)
    ]
    prompt_logits = pad_sequence(
        prompt_logits, batch_first=True, padding_value=IGNORE_ID
    )
    dist_logits = [
        last_hidden_state[i, text_token_len[i] + prompt_feats_len[i] :, :]
        for i in range(batch_size)
    ]
    dist_logits = pad_sequence(dist_logits, batch_first=True, padding_value=IGNORE_ID)
    dist_logits = dist_logits[:, : target_feats_len.max(), :]
    return text_logits, prompt_logits, dist_logits


def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    # If the input dimension is 2 or 3,
    # then we use ESPnet-ONNX based implementation for tracable modeling.
    # otherwise we use the traditional implementation for research use.
    if isinstance(lengths, list):
        logging.warning(
            "Using make_pad_mask with a list of lengths is not tracable. "
            + "If you try to trace this function with type(lengths) == list, "
            + "please change the type of lengths to torch.LongTensor."
        )

    if (
        (xs is None or xs.dim() in (2, 3))
        and length_dim <= 2
        and (not isinstance(lengths, list) and lengths.dim() == 1)
    ):
        return _make_pad_mask_traceable(lengths, xs, length_dim, maxlen)
    else:
        return _make_pad_mask(lengths, xs, length_dim, maxlen)


def _make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None, "When maxlen is specified, xs must not be specified."
        assert maxlen >= int(
            max(lengths)
        ), f"maxlen {maxlen} must be >= max(lengths) {max(lengths)}"

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert (
            xs.size(0) == bs
        ), f"The size of x.size(0) {xs.size(0)} must match the batch size {bs}"

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def _make_pad_mask_traceable(lengths, xs, length_dim, maxlen=None):
    """
    Make mask tensor containing indices of padded part.
    This is a simplified implementation of make_pad_mask without the xs input
    that supports JIT tracing for applications like exporting models to ONNX.
    Dimension length of xs should be 2 or 3
    This function will create torch.ones(maxlen, maxlen).triu(diagonal=1) and
    select rows to create mask tensor.
    """

    if xs is None:
        device = lengths.device
    else:
        device = xs.device

    if xs is not None and len(xs.shape) == 3:
        if length_dim == 1:
            lengths = lengths.unsqueeze(1).expand(*xs.transpose(1, 2).shape[:2])
        else:
            # Then length_dim is 2 or -1.
            if length_dim not in (-1, 2):
                logging.warn(
                    f"Invalid length_dim {length_dim}."
                    + "We set it to -1, which is the default value."
                )
                length_dim = -1
            lengths = lengths.unsqueeze(1).expand(*xs.shape[:2])

    if maxlen is not None:
        assert xs is None
        assert maxlen >= lengths.max()
    elif xs is not None:
        maxlen = xs.shape[length_dim]
    else:
        maxlen = lengths.max()

    # clip max(length) to maxlen
    lengths = torch.clamp(lengths, max=maxlen).type(torch.long)

    mask = torch.ones(maxlen + 1, maxlen + 1, dtype=torch.bool, device=device)
    mask = triu_onnx(mask)[1:, :-1]  # onnx cannot handle diagonal argument.
    mask = mask[lengths - 1][..., :maxlen]

    if xs is not None and len(xs.shape) == 3 and length_dim == 1:
        return mask.transpose(1, 2)
    else:
        return mask


def triu_onnx(x):
    arange = torch.arange(x.size(0), device=x.device)
    mask = arange.unsqueeze(-1).expand(-1, x.size(0)) <= arange
    return x * mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)


def mask_by_length(xs, lengths, fill=0):
    """Mask tensor according to length.

    Args:
        xs (Tensor): Batch of input tensor (B, `*`).
        lengths (LongTensor or List): Batch of lengths (B,).
        fill (int or float): Value to fill masked part.

    Returns:
        Tensor: Batch of masked input tensor (B, `*`).

    Examples:
        >>> x = torch.arange(5).repeat(3, 1) + 1
        >>> x
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]])
        >>> lengths = [5, 3, 2]
        >>> mask_by_length(x, lengths)
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 0, 0],
                [1, 2, 0, 0, 0]])

    """
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def to_torch_tensor(x):
    """Change to torch.Tensor or ComplexTensor from numpy.ndarray.

    Args:
        x: Inputs. It should be one of numpy.ndarray, Tensor, ComplexTensor, and dict.

    Returns:
        Tensor or ComplexTensor: Type converted inputs.

    Examples:
        >>> xs = np.ones(3, dtype=np.float32)
        >>> xs = to_torch_tensor(xs)
        tensor([1., 1., 1.])
        >>> xs = torch.ones(3, 4, 5)
        >>> assert to_torch_tensor(xs) is xs
        >>> xs = {'real': xs, 'imag': xs}
        >>> to_torch_tensor(xs)
        ComplexTensor(
        Real:
        tensor([1., 1., 1.])
        Imag;
        tensor([1., 1., 1.])
        )

    """
    # If numpy, change to torch tensor
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "c":
            # Dynamically importing because torch_complex requires python3
            from torch_complex.tensor import ComplexTensor

            return ComplexTensor(x)
        else:
            return torch.from_numpy(x)

    # If {'real': ..., 'imag': ...}, convert to ComplexTensor
    elif isinstance(x, dict):
        # Dynamically importing because torch_complex requires python3
        from torch_complex.tensor import ComplexTensor

        if "real" not in x or "imag" not in x:
            raise ValueError("has 'real' and 'imag' keys: {}".format(list(x)))
        # Relative importing because of using python3 syntax
        return ComplexTensor(x["real"], x["imag"])

    # If torch.Tensor, as it is
    elif isinstance(x, torch.Tensor):
        return x

    else:
        error = (
            "x must be numpy.ndarray, torch.Tensor or a dict like "
            "{{'real': torch.Tensor, 'imag': torch.Tensor}}, "
            "but got {}".format(type(x))
        )
        try:
            from torch_complex.tensor import ComplexTensor
        except Exception:
            # If PY2
            raise ValueError(error)
        else:
            # If PY3
            if isinstance(x, ComplexTensor):
                return x
            else:
                raise ValueError(error)


def get_subsample(train_args, mode, arch):
    """Parse the subsampling factors from the args for the specified `mode` and `arch`.

    Args:
        train_args: argument Namespace containing options.
        mode: one of ('asr', 'mt', 'st')
        arch: one of ('rnn', 'rnn-t', 'rnn_mix', 'rnn_mulenc', 'transformer')

    Returns:
        np.ndarray / List[np.ndarray]: subsampling factors.
    """
    if arch == "transformer":
        return np.array([1])

    elif mode == "mt" and arch == "rnn":
        # +1 means input (+1) and layers outputs (train_args.elayer)
        subsample = np.ones(train_args.elayers + 1, dtype=np.int64)
        logging.warning("Subsampling is not performed for machine translation.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif (
        (mode == "asr" and arch in ("rnn", "rnn-t"))
        or (mode == "mt" and arch == "rnn")
        or (mode == "st" and arch == "rnn")
    ):
        subsample = np.ones(train_args.elayers + 1, dtype=np.int64)
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(min(train_args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                "Subsampling is not performed for vgg*. "
                "It is performed in max pooling layers at CNN."
            )
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mix":
        subsample = np.ones(
            train_args.elayers_sd + train_args.elayers + 1, dtype=np.int64
        )
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(
                min(train_args.elayers_sd + train_args.elayers + 1, len(ss))
            ):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                "Subsampling is not performed for vgg*. "
                "It is performed in max pooling layers at CNN."
            )
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mulenc":
        subsample_list = []
        for idx in range(train_args.num_encs):
            subsample = np.ones(train_args.elayers[idx] + 1, dtype=np.int64)
            if train_args.etype[idx].endswith("p") and not train_args.etype[
                idx
            ].startswith("vgg"):
                ss = train_args.subsample[idx].split("_")
                for j in range(min(train_args.elayers[idx] + 1, len(ss))):
                    subsample[j] = int(ss[j])
            else:
                logging.warning(
                    "Encoder %d: Subsampling is not performed for vgg*. "
                    "It is performed in max pooling layers at CNN.",
                    idx + 1,
                )
            logging.info("subsample: " + " ".join([str(x) for x in subsample]))
            subsample_list.append(subsample)
        return subsample_list

    else:
        raise ValueError("Invalid options: mode={}, arch={}".format(mode, arch))


def rename_state_dict(
    old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]
):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from espnet.nets.pytorch_backend.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()


def trim_by_ctc_posterior(
    h: torch.Tensor,
    ctc_probs: torch.Tensor,
    masks: torch.Tensor,
    pos_emb: torch.Tensor = None,
):
    """
    Trim the encoder hidden output using CTC posterior.
    The continuous frames in the tail that confidently represent
    blank symbols are trimmed.
    """

    # Empirical settings
    frame_tolerance = 5
    conf_tolerance = 0.95
    blank_id = 0

    assert masks.size(1) == 1
    masks = masks.squeeze(1)
    hlens = masks.sum(dim=1)
    assert h.size()[:2] == ctc_probs.size()[:2]
    assert h.size(0) == hlens.size(0)

    # blank frames
    max_values, max_indices = ctc_probs.max(dim=2)
    blank_masks = torch.logical_and(
        max_values > conf_tolerance, max_indices == blank_id
    )

    # plus ignored frames
    joint_masks = torch.logical_or(blank_masks, ~masks)

    # lengths after the trimming
    B, T, _ = h.size()
    frame_idx = torch.where(
        joint_masks, -1, torch.arange(T).unsqueeze(0).repeat(B, 1).to(h.device)
    )
    after_lens = torch.where(
        frame_idx.max(dim=-1)[0] + frame_tolerance + 1 < hlens,
        frame_idx.max(dim=-1)[0] + frame_tolerance + 1,
        hlens,
    )

    h = h[:, : max(after_lens)]
    masks = ~make_pad_mask(after_lens).to(h.device).unsqueeze(1)

    if pos_emb is None:
        pos_emb = None
    elif (hlens.max() * 2 - 1).item() == pos_emb.size(1):  # RelPositionalEncoding
        pos_emb = pos_emb[
            :, pos_emb.size(1) // 2 - h.size(1) + 1 : pos_emb.size(1) // 2 + h.size(1)
        ]
    else:
        pos_emb = pos_emb[:, : h.size(1)]

    return h, masks, pos_emb
