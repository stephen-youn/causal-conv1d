import torch
import triton
import triton.language as tl

@triton.jit
def _causal_conv1d_fw(
    X, W, B, OUT,
    seqlen, width,
    stride_x_batch, stride_x_seqlen, stride_x_dim,
    stride_w_dim,
    stride_out_batch, stride_out_seqlen, stride_out_dim,
    BLOCK_N: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)
    n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < seqlen

    x_ptr_base = X + b * stride_x_batch + d * stride_x_dim
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(width):
        k_rev = width - 1 - k
        w_val = tl.load(W + d * stride_w_dim + k_rev)
        x_ptrs = x_ptr_base + (n - k) * stride_x_seqlen
        x_val = tl.load(x_ptrs, mask=mask_n & (n >= k), other=0.0)
        acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    if HAS_BIAS:
        acc += tl.load(B + d)

    out_ptrs = OUT + b * stride_out_batch + n * stride_out_seqlen + d * stride_out_dim
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_n)


def causal_conv1d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Depthwise causal conv1d implemented in Triton.

    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,) optional
    Returns:
        out: (batch, dim, seqlen)
    """
    assert x.is_cuda and weight.is_cuda
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    x_cl = x.transpose(1, 2).contiguous()  # (batch, seqlen, dim)
    out = torch.empty_like(x_cl)
    BLOCK_N = 128
    grid = (batch, dim, triton.cdiv(seqlen, BLOCK_N))
    bias_ptr = bias if bias is not None else x.new_empty(1)
    with torch.cuda.device(x.device):
        _causal_conv1d_fw[grid](
            x_cl, weight, bias_ptr, out,
            seqlen, width,
            x_cl.stride(0), x_cl.stride(1), x_cl.stride(2),
            weight.stride(0),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_N=BLOCK_N, HAS_BIAS=bias is not None
        )
    return out.transpose(1, 2)
