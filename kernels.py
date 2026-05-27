"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.

    TODO: implement.
    """
    rr = tl.dot(a_re, b_re, out_dtype=tl.float32)
    ii = tl.dot(a_im, b_im, out_dtype=tl.float32)
    ri = tl.dot(a_re, b_im, out_dtype=tl.float32)
    ir = tl.dot(a_im, b_re, out_dtype=tl.float32)

    return rr - ii, ri + ir


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    assert N >= 2 and (N & (N - 1)) == 0, f"N must be a power of 2 >= 2; got {N}"

    k = N.bit_length() - 1

    n256, rem = divmod(k, 8)
    n16, rem = divmod(rem, 4)

    chunks = [256] * n256
    chunks += [16] * n16

    if rem:
        chunks.append(1 << rem)

    assert math.prod(chunks) == N
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.

    TODO: implement.
    """
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)

    row_ids = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_ids = program_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, N, BLOCK_K):
        output_ids = k0 + tl.arange(0, BLOCK_K)

        # input tile
        input_mask = (row_ids[:, None] < B) & (output_ids[None, :] < N)

        x_re = tl.load(
            x_re_ptr + row_ids[:, None] * N + output_ids[None, :],
            mask=input_mask,
            other=0.0,
        )

        x_im = tl.load(
            x_im_ptr + row_ids[:, None] * N + output_ids[None, :],
            mask=input_mask,
            other=0.0,
        )

        # W is stored as W[n,k], but tl.dot wants it as W_T[k,n]
        matrix_mask = (output_ids[:, None] < N) & (input_ids[None, :] < N)

        W_re_T = tl.load(
            W_re_ptr + input_ids[None, :] * N + output_ids[:, None],
            mask=matrix_mask,
            other=0.0,
        )

        W_im_T = tl.load(
            W_im_ptr + input_ids[None, :] * N + output_ids[:, None],
            mask=matrix_mask,
            other=0.0,
        )

        part_re, part_im = _cdot(x_re, x_im, W_re_T, W_im_T)

        acc_re += part_re
        acc_im += part_im

    output_mask = (row_ids[:, None] < B) & (input_ids[None, :] < N)

    tl.store(
        y_re_ptr + row_ids[:, None] * N + input_ids[None, :],
        acc_re,
        mask=output_mask,
    )

    tl.store(
        y_im_ptr + row_ids[:, None] * N + input_ids[None, :],
        acc_im,
        mask=output_mask,
    )


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.

    TODO: implement.
    """
    B, N = x_re.shape

    BLOCK_M = 16
    BLOCK_K = 32
    BLOCK_N = 16

    grid = (
        triton.cdiv(B, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    f1_kernel[grid](
        x_re, x_im,
        W_re, W_im,
        y_re, y_im,
        B,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.

    TODO: implement.
    """
    program_id = tl.program_id(0)

    offs = tl.arange(0, N)

    # load the input in bit-reversed order
    perm = tl.load(perm_ptr + offs)

    v_re = tl.load(x_re_ptr + program_id * N + perm)
    v_im = tl.load(x_im_ptr + program_id * N + perm)

    # run each butterfly level
    for s in tl.static_range(0, LOG2_N):
        half = 1 << s
        span = half << 1

        pos = offs & (span - 1)
        is_low = pos < half

        lo_idx = tl.where(is_low, offs, offs - half)
        hi_idx = lo_idx + half

        lo_re = v_re.gather(lo_idx, axis=0)
        lo_im = v_im.gather(lo_idx, axis=0)

        hi_re = v_re.gather(hi_idx, axis=0)
        hi_im = v_im.gather(hi_idx, axis=0)

        # index into the radix-2 twiddle table
        tw_idx = (lo_idx & (half - 1)) * (N >> (s + 1))

        w_re = tl.load(tw_re_ptr + tw_idx)
        w_im = tl.load(tw_im_ptr + tw_idx)

        # multiply the top half by the twiddle
        t_re = w_re * hi_re - w_im * hi_im
        t_im = w_re * hi_im + w_im * hi_re

        out_low_re = lo_re + t_re
        out_low_im = lo_im + t_im

        out_high_re = lo_re - t_re
        out_high_im = lo_im - t_im

        v_re = tl.where(is_low, out_low_re, out_high_re)
        v_im = tl.where(is_low, out_low_im, out_high_im)

    # F3 uses this to apply the Bailey twiddle after the first small FFT
    if BAILEY_EPILOGUE:
        outer = program_id % OUTER_DIM

        bt_re = tl.load(bt_re_ptr + outer * N + offs)
        bt_im = tl.load(bt_im_ptr + outer * N + offs)

        new_re = v_re * bt_re - v_im * bt_im
        new_im = v_re * bt_im + v_im * bt_re

        v_re = new_re
        v_im = new_im

    # write the row back out
    if STRIDED_STORE:
        # F3-B stores directly in final transposed order
        b = program_id // OUTER_DIM
        k2 = program_id - b * OUTER_DIM

        out_offsets = b * N_TOTAL + offs * OUTER_DIM + k2

        tl.store(y_re_ptr + out_offsets, v_re)
        tl.store(y_im_ptr + out_offsets, v_im)

    else:
        tl.store(y_re_ptr + program_id * N + offs, v_re)
        tl.store(y_im_ptr + program_id * N + offs, v_im)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode.

    TODO: implement.
    """
    B, N = x_re.shape

    assert N >= 2 and (N & (N - 1)) == 0
    LOG2_N = int(math.log2(N))

    grid = (B,)

    f2_kernel[grid](
        x_re, x_im,
        y_re, y_im,
        tw_re, tw_im,
        perm,
        tw_re, tw_im,      # dummy Bailey pointers; unused for vanilla F2
        1, 0,              # OUTER_DIM, N_TOTAL unused here
        N=N,
        LOG2_N=LOG2_N,
        BAILEY_EPILOGUE=False,
        STRIDED_STORE=False,
        num_warps=8,
        num_stages=1,
    )


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.

    TODO: implement.
    """
    program_r = tl.program_id(0)
    program_c = tl.program_id(1)
    program_b = tl.program_id(2)

    r_ids = program_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c_ids = program_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask = (r_ids[:, None] < R) & (c_ids[None, :] < C)

    # input is x[b,r,c]
    in_offsets = program_b * R * C + r_ids[:, None] * C + c_ids[None, :]

    # output is y[b,c,r]
    out_offsets = program_b * R * C + c_ids[None, :] * R + r_ids[:, None]

    x_re = tl.load(x_re_ptr + in_offsets, mask=mask, other=0.0)
    x_im = tl.load(x_im_ptr + in_offsets, mask=mask, other=0.0)

    tl.store(y_re_ptr + out_offsets, x_re, mask=mask)
    tl.store(y_im_ptr + out_offsets, x_im, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.

    TODO: implement.
    """
    program_id = tl.program_id(0)

    # each program handles a few batch rows
    local_rows = tl.arange(0, BLOCK_B * 16)[:, None]
    cols16 = tl.arange(0, 16)[None, :]

    bb = local_rows // 16
    d1 = local_rows - bb * 16
    d0 = cols16

    b = program_id * BLOCK_B + bb
    row_mask = b < B

    # load the 16 point DFT matrix
    f_r = tl.arange(0, 16)[:, None]
    f_c = tl.arange(0, 16)[None, :]

    F_re = tl.load(F_re_ptr + f_r * 16 + f_c)
    F_im = tl.load(F_im_ptr + f_r * 16 + f_c)

    # Stage 0:
    # first radix-16 stage: FFT over d0 for each fixed d1
    x_offsets = b * 256 + d0 * 16 + d1

    a0_re = tl.load(x_re_ptr + x_offsets, mask=row_mask, other=0.0)
    a0_im = tl.load(x_im_ptr + x_offsets, mask=row_mask, other=0.0)

    s0_re, s0_im = _cdot(a0_re, a0_im, F_re, F_im)

    # s=1 sanity-check path:
    # sanity check path only does the first stage
    if STAGE_STOP == 1:
        e1 = cols16

        out_offsets_s1 = b * 256 + e1 * 16 + d1

        tl.store(y_re_ptr + out_offsets_s1, s0_re.to(tl.float16), mask=row_mask)
        tl.store(y_im_ptr + out_offsets_s1, s0_im.to(tl.float16), mask=row_mask)
        return

    # Stage 1:
    # move e1 in front so the next FFT runs over d1
    t_re = s0_re.to(tl.float16).reshape((BLOCK_B, 16, 16))
    t_im = s0_im.to(tl.float16).reshape((BLOCK_B, 16, 16))

    t_re = tl.permute(t_re, (0, 2, 1))
    t_im = tl.permute(t_im, (0, 2, 1))

    a1_re = t_re.reshape((BLOCK_B * 16, 16))
    a1_im = t_im.reshape((BLOCK_B * 16, 16))

    # rows are now (b,e1), columns are d1
    rows1 = tl.arange(0, BLOCK_B * 16)[:, None]
    d1_cols = tl.arange(0, 16)[None, :]

    bb1 = rows1 // 16
    e1_row = rows1 - bb1 * 16
    b1 = program_id * BLOCK_B + bb1
    row_mask1 = b1 < B

    # stage 1 twiddle
    tw_offs = 1 * 16 * 16 + d1_cols * 16 + e1_row

    tw_re = tl.load(tw_re_ptr + tw_offs)
    tw_im = tl.load(tw_im_ptr + tw_offs)

    a1_tw_re = a1_re * tw_re - a1_im * tw_im
    a1_tw_im = a1_re * tw_im + a1_im * tw_re

    # second radix-16 stage: FFT over d1
    s1_re, s1_im = _cdot(
        a1_tw_re.to(tl.float16),
        a1_tw_im.to(tl.float16),
        F_re,
        F_im,
    )

    # Final output:
    # final natural index is k = e0*16 + e1
    e0 = cols16
    k = e0 * 16 + e1_row

    if STORE_T:
        # F7 wants the final transpose fused into this store
        outer = b1 // M
        inner = b1 - outer * M

        out_offsets = outer * 256 * M + k * M + inner
    else:
        out_offsets = b1 * 256 + k

    tl.store(y_re_ptr + out_offsets, s1_re.to(tl.float16), mask=row_mask1)
    tl.store(y_im_ptr + out_offsets, s1_im.to(tl.float16), mask=row_mask1)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.

    TODO: implement.
    """
    program_id = tl.program_id(0)

    row_ids = program_id * BLOCK_B + tl.arange(0, BLOCK_B)
    input_ids = tl.arange(0, 16)

    mask_x = (row_ids[:, None] < rows) & (input_ids[None, :] < R)

    # zero pad the small input up to 16 columns
    x_re = tl.load(
        x_re_ptr + row_ids[:, None] * R + input_ids[None, :],
        mask=mask_x,
        other=0.0,
    )
    x_im = tl.load(
        x_im_ptr + row_ids[:, None] * R + input_ids[None, :],
        mask=mask_x,
        other=0.0,
    )

    # load the transpose of the small DFT matrix
    r = tl.arange(0, 16)[:, None]
    c = tl.arange(0, 16)[None, :]

    MT_re = tl.load(M_re_ptr + c * 16 + r)
    MT_im = tl.load(M_im_ptr + c * 16 + r)

    out_re, out_im = _cdot(x_re, x_im, MT_re, MT_im)

    output_ids = tl.arange(0, 16)
    mask_y = (row_ids[:, None] < rows) & (output_ids[None, :] < R)

    if STORE_T:
        # rows = rows_outer * M
        # input row index = outer*M + inner
        outer = row_ids // M
        inner = row_ids - outer * M

        # output layout: (rows_outer, R, M)
        out_offsets = outer[:, None] * R * M + output_ids[None, :] * M + inner[:, None]
    else:
        # normal row-major output
        out_offsets = row_ids[:, None] * R + output_ids[None, :]

    tl.store(y_re_ptr + out_offsets, out_re.to(tl.float16), mask=mask_y)
    tl.store(y_im_ptr + out_offsets, out_im.to(tl.float16), mask=mask_y)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).

    TODO: implement.
    """
    program_m0 = tl.program_id(0)
    program_M = tl.program_id(1)
    program_row = tl.program_id(2)

    row_ids0 = program_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    M_ids = program_M * BLOCK_M + tl.arange(0, BLOCK_M)

    mask = (row_ids0[:, None] < m0) & (M_ids[None, :] < M)

    # input is x[row,n1,kM]
    in_offsets = (
        program_row * m0 * M
        + row_ids0[:, None] * M
        + M_ids[None, :]
    )

    tw_offsets = row_ids0[:, None] * M + M_ids[None, :]

    x_re = tl.load(x_re_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    x_im = tl.load(x_im_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)

    tw_re = tl.load(tw_re_ptr + tw_offsets, mask=mask, other=0.0).to(tl.float32)
    tw_im = tl.load(tw_im_ptr + tw_offsets, mask=mask, other=0.0).to(tl.float32)

    y_re = x_re * tw_re - x_im * tw_im
    y_im = x_re * tw_im + x_im * tw_re

    if STORE_T:
        # fused transpose output: y[row,kM,n1]
        out_offsets = (
            program_row * M * m0
            + M_ids[None, :] * m0
            + row_ids0[:, None]
        )
    else:
        # keep the same layout
        out_offsets = in_offsets

    tl.store(y_re_ptr + out_offsets, y_re.to(tl.float16), mask=mask)
    tl.store(y_im_ptr + out_offsets, y_im.to(tl.float16), mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store

    TODO: implement.
    """
    N1 = plan['N1']
    N2 = plan['N2']
    N = plan['N']

    # 1. T1:
    # input view:  (B, N2, N1)
    # output view: (B, N1, N2)
    # in -> mid
    _transpose(
        in_re, in_im,
        mid_re, mid_im,
        B, N2, N1,
    )

    # 2. F2-A:
    # length-N2 FFT over B*N1 rows.
    # Also multiply by Bailey twiddle bt[n1, k2].
    # mid -> out
    f2_kernel[(B * N1,)](
        mid_re, mid_im,
        out_re, out_im,
        plan['tw_re_n2'], plan['tw_im_n2'],
        plan['perm_n2'],
        plan['bt_re'], plan['bt_im'],
        N1, N,
        N=N2,
        LOG2_N=plan['LOG2_N2'],
        BAILEY_EPILOGUE=True,
        STRIDED_STORE=False,
        num_warps=8,
        num_stages=1,
    )

    # 3. T2:
    # input view:  (B, N1, N2)
    # output view: (B, N2, N1)
    # out -> mid
    _transpose(
        out_re, out_im,
        mid_re, mid_im,
        B, N1, N2,
    )

    # 4. F2-B:
    # length-N1 FFT over B*N2 rows.
    # STRIDED_STORE=True writes final output as (B, N1, N2),
    # i.e. y[b, k1, k2] = out[b*N + k1*N2 + k2].
    # mid -> out
    f2_kernel[(B * N2,)](
        mid_re, mid_im,
        out_re, out_im,
        plan['tw_re_n1'], plan['tw_im_n1'],
        plan['perm_n1'],
        plan['tw_re_n1'], plan['tw_im_n1'],   # dummy bt pointers
        N2, N,
        N=N1,
        LOG2_N=plan['LOG2_N1'],
        BAILEY_EPILOGUE=False,
        STRIDED_STORE=True,
        num_warps=8,
        num_stages=1,
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)

    TODO: implement.
    """
    N1 = plan['N1']   # 256
    N2 = plan['N2']   # 256

    # 1. T1:
    # input:  (B, N2, N1)
    # output: (B, N1, N2)
    # in -> b0
    _transpose(
        in_re, in_im,
        b0_re, b0_im,
        B, N2, N1,
    )

    # 2. FFT-A:
    # length-256 FFT over the last axis of (B, N1, N2)
    # This is B*N1 separate length-256 signals.
    # b0 -> b1
    _fft_chunk(
        b0_re, b0_im,
        b1_re, b1_im,
        rows=B * N1,
        m=N2,
        plan=plan,
    )

    # 3. Scale:
    # logical view: (B, N1, N2)
    # multiply by bt[n1, k2]
    # b1 -> b0
    _scale(
        b1_re, b1_im,
        b0_re, b0_im,
        rows=B,
        m0=N1,
        M=N2,
        twr=plan['bt_re'],
        twi=plan['bt_im'],
        store_t=False,
    )

    # 4. T2:
    # input:  (B, N1, N2)
    # output: (B, N2, N1)
    # b0 -> b1
    _transpose(
        b0_re, b0_im,
        b1_re, b1_im,
        B, N1, N2,
    )

    # 5. FFT-B:
    # length-256 FFT over the last axis of (B, N2, N1)
    # This is B*N2 separate length-256 signals.
    # b1 -> b2
    _fft_chunk(
        b1_re, b1_im,
        b2_re, b2_im,
        rows=B * N2,
        m=N1,
        plan=plan,
    )

    # 6. T3:
    # input:  (B, N2, N1)
    # output: (B, N1, N2)
    # b2 -> b0 final
    _transpose(
        b2_re, b2_im,
        b0_re, b0_im,
        B, N2, N1,
    )


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.

    TODO: implement.
    """
    # Total FFT length at this recursion level.
    N_i = math.prod(chunks)

    # Leaf:
    # just do one chunk FFT over (rows, m0).
    if len(chunks) == 1:
        m0 = chunks[0]

        out_re, out_im = cyc.next()

        _fft_chunk(
            cur_re, cur_im,
            out_re, out_im,
            rows=rows,
            m=m0,
            plan=plan,
        )

        return out_re, out_im

    # Non-leaf Bailey split.
    # chunks = [m0, ...]
    # current logical shape: (rows, M, m0)
    m0 = chunks[0]
    rest = chunks[1:]
    M = math.prod(rest)

    # 1. T1:
    # (rows, M, m0) -> (rows, m0, M)
    # cur -> t1
    t1_re, t1_im = cyc.next()

    _transpose(
        cur_re, cur_im,
        t1_re, t1_im,
        rows,
        M,
        m0,
    )

    # 2. Recurse:
    # length-M FFT over rows*m0 signals.
    # input/output logical shape: (rows*m0, M)
    rec_re, rec_im = _f6_rec(
        t1_re, t1_im,
        rows * m0,
        rest,
        plan,
        cyc,
    )

    # 3. Scale:
    # logical shape: (rows, m0, M)
    # multiply by w_N_i^(n1*kM)
    # rec -> scaled
    scaled_re, scaled_im = cyc.next()

    twr, twi = _lookup_tw(plan, m0, M, N_i)

    _scale(
        rec_re, rec_im,
        scaled_re, scaled_im,
        rows,
        m0,
        M,
        twr,
        twi,
        store_t=False,
    )

    # 4. T2:
    # (rows, m0, M) -> (rows, M, m0)
    # scaled -> t2
    t2_re, t2_im = cyc.next()

    _transpose(
        scaled_re, scaled_im,
        t2_re, t2_im,
        rows,
        m0,
        M,
    )

    # 5. FFT-m0:
    # length-m0 FFT over rows*M signals.
    # t2 -> fft0
    fft0_re, fft0_im = cyc.next()

    _fft_chunk(
        t2_re, t2_im,
        fft0_re, fft0_im,
        rows=rows * M,
        m=m0,
        plan=plan,
    )

    # 6. T3:
    # (rows, M, m0) -> (rows, m0, M)
    # fft0 -> out
    out_re, out_im = cyc.next()

    _transpose(
        fft0_re, fft0_im,
        out_re, out_im,
        rows,
        M,
        m0,
    )

    return out_re, out_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.

    TODO: implement.
    """
    N_i = math.prod(chunks)

    # Leaf:
    # Just do one chunk FFT over (rows, m0).
    # No fusion needed here because there is no outer T3 at this level.
    if len(chunks) == 1:
        m0 = chunks[0]

        out_re, out_im = cyc.next()

        _fft_chunk(
            cur_re, cur_im,
            out_re, out_im,
            rows=rows,
            m=m0,
            plan=plan,
        )

        return out_re, out_im

    # Non-leaf Bailey split.
    # Current logical shape: (rows, M, m0)
    m0 = chunks[0]
    rest = chunks[1:]
    M = math.prod(rest)

    # 1. T1:
    # (rows, M, m0) -> (rows, m0, M)
    # cur -> t1
    t1_re, t1_im = cyc.next()

    _transpose(
        cur_re, cur_im,
        t1_re, t1_im,
        rows,
        M,
        m0,
    )

    # 2. Recurse:
    # length-M FFT over rows*m0 signals.
    # Input/output logical shape: (rows*m0, M),
    # equivalently (rows, m0, M).
    rec_re, rec_im = _f7_rec(
        t1_re, t1_im,
        rows * m0,
        rest,
        plan,
        cyc,
    )

    # 3. Scale + T2 fused:
    # Input logical shape:  (rows, m0, M)
    # Output logical shape: (rows, M, m0)
    # This replaces:
    #   Scale -> T2
    scaled_t_re, scaled_t_im = cyc.next()

    twr, twi = _lookup_tw(plan, m0, M, N_i)

    _scale(
        rec_re, rec_im,
        scaled_t_re, scaled_t_im,
        rows,
        m0,
        M,
        twr,
        twi,
        store_t=True,
    )

    # 4. FFT-m0 + T3 fused:
    # Input logical shape:  (rows, M, m0)
    # Treat as rows*M separate length-m0 FFTs.
    # STORE_T=True writes output as:
    #   (rows, m0, M)
    # This replaces:
    #   FFT-m0 -> T3
    out_re, out_im = cyc.next()

    _fft_chunk(
        scaled_t_re, scaled_t_im,
        out_re, out_im,
        rows=rows * M,
        m=m0,
        plan=plan,
        M=M,
        store_t=True,
    )

    return out_re, out_im

