import torch
import triton
import triton.language as tl

@triton.jit
def _causal_conv1d_fw(
    X, W, B, OUT, PRE,
    seqlen, width,
    stride_x_batch, stride_x_seqlen, stride_x_dim,
    stride_w_dim,
    stride_out_batch, stride_out_seqlen, stride_out_dim,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SAVE_PRE: tl.constexpr,
    DO_ACT: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)
    n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < seqlen

    x_ptr_base = X + b * stride_x_batch + d * stride_x_dim
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(width):
        w_val = tl.load(W + d * stride_w_dim + k)
        x_ptrs = x_ptr_base + (n - k) * stride_x_seqlen
        x_val = tl.load(x_ptrs, mask=mask_n & (n >= k), other=0.0)
        acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    if HAS_BIAS:
        acc += tl.load(B + d)

    if SAVE_PRE:
        pre_ptrs = PRE + b * stride_out_batch + n * stride_out_seqlen + d * stride_out_dim
        tl.store(pre_ptrs, acc, mask=mask_n)

    if DO_ACT:
        acc = acc * tl.sigmoid(acc)

    out_ptrs = OUT + b * stride_out_batch + n * stride_out_seqlen + d * stride_out_dim
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_n)


@triton.jit
def _causal_conv1d_bw_dx(
    DOUT, W, DX,
    seqlen, width,
    stride_dout_batch, stride_dout_seqlen, stride_dout_dim,
    stride_w_dim,
    stride_dx_batch, stride_dx_seqlen, stride_dx_dim,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)
    n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < seqlen

    dout_ptr_base = DOUT + b * stride_dout_batch + d * stride_dout_dim
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(width):
        w_val = tl.load(W + d * stride_w_dim + k)
        dout_ptrs = dout_ptr_base + (n + k) * stride_dout_seqlen
        dout_val = tl.load(dout_ptrs, mask=mask_n & (n + k < seqlen), other=0.0)
        acc += dout_val.to(tl.float32) * w_val.to(tl.float32)

    out_ptrs = DX + b * stride_dx_batch + n * stride_dx_seqlen + d * stride_dx_dim
    tl.store(out_ptrs, acc.to(DX.dtype.element_ty), mask=mask_n)


def _causal_conv1d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    save_pre: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Triton causal conv1d forward kernel."""
    assert x.is_cuda and weight.is_cuda
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    x_cl = x.transpose(1, 2).contiguous()  # (batch, seqlen, dim)
    out = torch.empty_like(x_cl)
    pre = torch.empty_like(x_cl) if save_pre else x_cl
    BLOCK_N = 128
    grid = (batch, dim, triton.cdiv(seqlen, BLOCK_N))
    bias_ptr = bias if bias is not None else x.new_empty(1)
    with torch.cuda.device(x.device):
        _causal_conv1d_fw[grid](
            x_cl,
            weight,
            bias_ptr,
            out,
            pre,
            seqlen,
            width,
            x_cl.stride(0),
            x_cl.stride(1),
            x_cl.stride(2),
            weight.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_N=BLOCK_N,
            HAS_BIAS=bias is not None,
            SAVE_PRE=save_pre,
            DO_ACT=activation == "silu",
        )
    if save_pre:
        pre = pre.transpose(1, 2)
    else:
        pre = None
    return out.transpose(1, 2), pre


def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Depthwise causal conv1d implemented in Triton with optional SiLU activation."""
    out, _ = _causal_conv1d_forward(x, weight, bias, activation, save_pre=False)
    return out


class CausalConv1dTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, activation=None):
        out, pre = _causal_conv1d_forward(x, weight, bias, activation, save_pre=activation == "silu")
        ctx.save_for_backward(x, weight, bias, pre)
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias, pre = ctx.saved_tensors
        grad = grad_out.contiguous()
        if ctx.activation == "silu":
            sig = torch.sigmoid(pre)
            grad = grad * (sig * (1 + pre * (1 - sig)))
        batch, dim, seqlen = grad.shape
        dx = torch.empty_like(grad)
        BLOCK_N = 128
        grid = (batch, dim, triton.cdiv(seqlen, BLOCK_N))
        with torch.cuda.device(grad.device):
            _causal_conv1d_bw_dx[grid](
                grad.transpose(1, 2),
                weight,
                dx.transpose(1, 2),
                seqlen,
                weight.shape[1],
                grad.transpose(1, 2).stride(0),
                grad.transpose(1, 2).stride(1),
                grad.transpose(1, 2).stride(2),
                weight.stride(0),
                dx.transpose(1, 2).stride(0),
                dx.transpose(1, 2).stride(1),
                dx.transpose(1, 2).stride(2),
                BLOCK_N=BLOCK_N,
            )
        dx = dx
        dweight = torch.zeros_like(weight)
        for k in range(weight.shape[1]):
            dweight[:, k] = (x[:, :, : seqlen - k] * grad[:, :, k:]).sum(dim=(0, 2))
        dbias = grad.sum(dim=(0, 2)) if bias is not None else None
        return dx, dweight, dbias, None


def causal_conv1d_triton_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    return CausalConv1dTritonFn.apply(x, weight, bias, activation)
