import torch
import pytest

from causal_conv1d.causal_conv1d_interface import (
    causal_conv1d_update,
    causal_conv1d_ref,
)
from causal_conv1d.causal_conv1d_triton import (
    causal_conv1d_update_triton,
    causal_conv1d_fn_triton,
    PAD_SLOT_ID,
)


def test_update_triton_matches_cpp():
    device = 'cuda'
    batch, dim, width, seqlen = 2, 32, 4, 8
    x = torch.randn(batch, dim, seqlen, device=device)
    conv_state_cpp = torch.randn(batch, dim, width - 1, device=device)
    conv_state_triton = conv_state_cpp.clone()
    weight = torch.randn(dim, width, device=device)
    bias = torch.randn(dim, device=device)

    out_cpp = causal_conv1d_update(x.clone(), conv_state_cpp, weight, bias)
    out_triton = causal_conv1d_update_triton(x.clone(), conv_state_triton, weight, bias)

    assert torch.allclose(out_cpp, out_triton, rtol=1e-3, atol=1e-3)
    assert torch.allclose(conv_state_cpp, conv_state_triton, rtol=1e-3, atol=1e-3)


def test_fn_triton_matches_ref():
    device = 'cuda'
    dim, width = 16, 4
    seqlens = [3, 2, 4]
    total = sum(seqlens)
    x_seqs = [torch.randn(1, dim, L, device=device) for L in seqlens]
    x_cat = torch.cat([s.squeeze(0) for s in x_seqs], dim=1)
    weight = torch.randn(dim, width, device=device)
    bias = torch.randn(dim, device=device)
    conv_states = torch.randn(len(seqlens), dim, width - 1, device=device)
    has_initial = torch.tensor([1, 0, 1], dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0).tolist()), dtype=torch.int32, device=device)
    cache_indices = torch.arange(len(seqlens), dtype=torch.int32, device=device)

    out_triton = causal_conv1d_fn_triton(
        x_cat,
        weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_states=has_initial,
        activation='silu',
        pad_slot_id=PAD_SLOT_ID,
    )

    out_ref_parts = []
    start = 0
    conv_states_ref = conv_states.clone()
    for i, L in enumerate(seqlens):
        x_seq = x_cat[:, start:start+L].unsqueeze(0)
        state = conv_states_ref[i:i+1] if has_initial[i] else None
        out, final_state = causal_conv1d_ref(
            x_seq,
            weight,
            bias,
            initial_states=state,
            return_final_states=True,
            activation='silu',
        )
        conv_states_ref[i] = final_state.squeeze(0)
        out_ref_parts.append(out.squeeze(0))
        start += L
    out_ref = torch.cat(out_ref_parts, dim=-1)

    assert torch.allclose(out_triton, out_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(conv_states, conv_states_ref, rtol=1e-3, atol=1e-3)
