# sme-jit-core Roadmap

## Where We Are

Gates 0–19 are complete. We have:
- A working JIT harness (MAP_JIT, fork isolation, GPR snapshots)
- Proof that M4 uses ARM SME (not AMX)
- A 16×16 SGEMM kernel (PTRUE → ZERO ZA → [LD1W×2 + FMOPA + ADD×2]×K → [ST1W + ADD×2]×16)
- Fused GEMM + Bias + ReLU kernels (Gate 17)
- Differential correctness: `max_diff = 0.0` vs Accelerate
- Benchmark: 1.8–2.5× faster than Accelerate at tile-sized problems
- A 3-layer MNIST inference engine running entirely through JIT'd SME kernels (Gate 18)

## Gate 16: BFMOPA / SMOPA Probing (Complete - Negative Result)

**Goal**: Discover which extended outer product instructions M4 supports.

**Status**: Completed.
**Findings**:
- M4 (macOS 15.x) **does not SIGILL** on `BFMOPA`, `BFMOPS`, `SMOPA`, `UMOPA`, or `SUMOPA`.
- However, all these instructions currently behave as **NOPs** (zero impact on ZA tile even with all-true predicates and valid SM/ZA state).
- Standard FP32 `FMOPA` remains the only functional outer product instruction.
- **Hypothesis**: The hardware has the decoding logic (hence no fault) but the execution pipelines for these optional SME features are either disabled in firmware or pending a microcode/OS update.

ARM SME defines several outer product variants beyond FMOPA (FP32):

| Instruction | Operands | MACs/inst | Potential speedup |
|:------------|:---------|:----------|:------------------|
| **BFMOPA** | BF16 → FP32 | 512 | 2× over FMOPA |
| **BFMOPS** | BF16 → FP32 (subtract) | 512 | — |
| **SMOPA** | INT8 → INT32 | 1024 | 4× over FMOPA |
| **UMOPA** | UINT8 → INT32 | 1024 | 4× over FMOPA |
| **SUMOPA** | INT8×UINT8 → INT32 | 1024 | 4× mixed |

**Approach**:
1. Encode each variant using known ARM encodings
2. Probe via fork-based harness (fault → not supported, no fault → supported)
3. For supported instructions, build correctness tests:
   - Pack BF16 inputs, execute BFMOPA, verify against FP32 reference
   - Pack INT8 inputs, execute SMOPA, verify against INT32 reference
4. Benchmark supported variants against Accelerate
5. Document M4 SME capability matrix (first public data)

**Key encodings to probe**:
- BFMOPA ZA0.S, P0/M, Z0.H, Z1.H: `0x8181_0000`
- SMOPA ZA0.S, P0/M, Z0.B, Z1.B: `0xA080_0000`
- UMOPA ZA0.S, P0/M, Z0.B, Z1.B: `0xA180_0000`

**Success criteria**: At least one new outer product instruction confirmed working, with correctness test and benchmark.

## Gate 17: Fused GEMM + Activation Kernels (Complete)

**Goal**: JIT-emit kernels that fuse matmul with activation functions in a single kernel — zero intermediate memory traffic.

**Status**: Complete. All three fusion variants pass with `max_diff = 0.0`.

**Results**:

| Variant | Instructions | Status | max_diff |
|:--------|:------------|:-------|:---------|
| GEMM + ReLU | 198 | ✅ PASS | 0.0 |
| GEMM + Bias | 198 | ✅ PASS | 0.0 |
| GEMM + Bias + ReLU | 215 | ✅ PASS | 0.0 |

**Architecture — Strategy C (Store-then-Modify)**:

The fusion uses a three-phase approach that avoids the unreliable MOVA ZA↔Z instruction:
1. **Phase 1**: FMOPA outer product loop → ZA accumulator (proven, Gate 14d)
2. **Phase 2**: ST1W ZA rows → output memory (proven, Gate 14d)
3. **Phase 3**: SVE LD1W row → apply activation → SVE ST1W back (data round-trips through L1)

Strategy B (in-place ZA fusion via MOVA) was abandoned after MOVA proved unreliable on M4.

**Key M4 SME discoveries**:
- `FMAX Zdn.S, Pg/M, Zdn.S, #0.0` (immediate form) is a **NOP** in streaming mode
- `FMAX Zdn.S, Pg/M, Zdn.S, Zm.S` (vector form with DUP Z4.S, #0) **works correctly**
- `FADD Zd.S, Zn.S, Zm.S` (unpredicated vector) **works correctly** in streaming mode
- SVE `LD1W` / `ST1W` (scalar+scalar) **work correctly** in streaming mode for Z registers

**Encodings confirmed working on M4**:
- `DUP Z4.S, #0` → `0x2538_C004`
- `FMAX Z2.S, P0/M, Z2.S, Z4.S` → `0x6586_8082`
- `FADD Z2.S, Z2.S, Z3.S` (unpredicated) → `0x6580_0062`
- `SVE ST1W {Z2.S}, P0, [X2, X3, LSL #2]` → `0xE543_4042`

## Gate 18: Tiny Inference Engine Demo (Complete)

**Goal**: Run a small neural network (2-3 layer MLP) entirely through JIT'd fused SME kernels.

**Status**: Complete. 16/16 correct, bit-exact match vs Accelerate reference.

**Architecture**: 784 → 16 (BiasReLU) → 16 (BiasReLU) → 10 (Bias, zero-padded to 16)

**Components**:
- `scripts/train_mnist.py` — trains MLP, exports weights as raw f32 binaries
- `src/weights.rs` — loads weight files, validates dimensions
- `src/inference.rs` — three inference paths:
  - `run_inference_probed()` — fork-isolated, safe for development
  - `run_inference_direct()` — direct JIT page calls, for benchmarking
  - `run_inference_reference()` — Accelerate-based, for differential testing

**Results**:

| Metric | Value |
|:-------|:------|
| Predictions correct | 16/16 |
| Hidden 1 max_diff | 0.00e0 |
| Hidden 2 max_diff | 0.00e0 |
| Output max_diff | 0.00e0 |
| Accelerate latency | 3.5 μs/batch |
| JIT direct latency | 33.6 μs/batch |

**Key insight**: The JIT path is ~10× slower than Accelerate for this workload because:
1. JIT pages are rebuilt every call (no caching yet)
2. The 784→16 layer requires a 16×784 transpose per batch
3. Accelerate's cblas_sgemm is hyper-optimized for rectangular shapes

This is a **correctness gate**, not a performance gate. The tile-sized (16×16) SGEMM
kernel still runs 1.8–2.5× faster than Accelerate (Gate 14d). The overhead here
is all in the orchestration layer — transposing, page construction, and the mismatch
between rectangular matmul (784→16) and square tile (16×16).

**Data layout protocol**:
- A (left matrix) must be stored **column-major** (transposed) for FMOPA
- B (right matrix) must be stored **row-major**
- K = number of FMOPA outer products = inner dimension of the matmul
- For 784→16: K=784, not K=49 (each FMOPA is one rank-1 update, not a 16-wide panel)

## Gate 19: Cached Inference Engine (Complete)

**Goal**: Eliminate per-call JIT page construction overhead to recover speed advantage.

**Status**: Complete. Correctness verified, 5× faster than uncached Gate 18.

**Architecture**:
- `CachedInferenceEngine` — builds 3 JIT pages once at construction time
- `build_sme_sgemm_page_cached()` — bakes immutable pointers (weights, bias), takes A/C via X0/X1
- `call_with_args(a_ptr, c_ptr)` — new JitPage method for register-based calling convention
- Pre-allocated buffers reused across calls (zero heap allocation per inference)

**Calling convention** (cached kernels):
- `X0` = A pointer (input, column-major) — passed at call time
- `X1` = C pointer (output, row-major) — passed at call time
- `X5` = B pointer (weights) — baked into instruction stream
- `X6` = Bias pointer — baked into instruction stream
- Kernel does: `MOV X4,X0; MOV X2,X1` then runs standard FMOPA pipeline

**Results**:

| Metric | Gate 18 (uncached) | Gate 19 (cached) |
|:-------|:-------------------|:-----------------|
| Build time | ~15 μs/call × 3 | 32.5 μs one-time |
| Inference latency | 33.6 μs/batch | 6.8 μs/batch |
| vs Accelerate | 0.10× | 0.46× |
| Correctness | 16/16, 0.00e0 | 16/16, 0.00e0 |

**Remaining bottleneck**: The input transpose (784×16 = 12,544 scalar copies) and 3× SMSTART/SMSTOP
per inference dominate the 6.8 μs. The FMOPA compute itself is ~3 μs.

**Next optimization opportunities** (for a future gate):
1. Pre-transpose input on the Python/data side → eliminates 784×16 transpose
2. Emit column-major ST1W in kernel → eliminates inter-layer 16×16 transposes
3. Single SMSTART/SMSTOP wrapping all 3 layers → eliminates 4 mode switches

## Deferred (from original roadmap)

These items are deprioritized but not abandoned:

| Item | Original Gate | Status |
|:-----|:-------------|:-------|
| Multi-tile tiling (M,N > 16) | Gate 16 (old) | Deferred — focus on depth before breadth |
| OS scheduler bypass (P-core pinning) | Gate 17 (old) | Deferred — nice for benchmarks, not critical path |
| Double-buffered loads | Gate 18 (old) | Deferred — optimization after new instruction discovery |
| Batched small-SGEMM | — | Deferred — build after fused kernels prove out |
| Pre-transposed data layout | — | Next priority — eliminates 784×16 transpose |
| Single SM session across layers | — | Next priority — eliminates 4 mode switches |

## Non-Goals (Archived)

- ~~AMX instruction encoding~~ — dead on M4
- ~~Frida heist scripts~~ — exploration complete, data preserved in git history
- ~~Planning documents~~ — replaced by this roadmap
