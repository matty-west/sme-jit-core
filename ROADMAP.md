# sme-jit-core Roadmap

## Where We Are

Gates 0–24, 26–27 are complete. We have transitioned from the **Discovery Phase** (empirical probing) to the **Maturation Phase** (API stability and architectural scaling).

### Status Summary
- **Proven Performance:** 1.55× faster than Accelerate for full MNIST MLPs; 5.1× for 16×16 tiles.
- **Architectural Milestone:** Successfully implemented monolithic kernel fusion and zero-transpose vertical stores.
- **Codebase Maturation (Active):**
    - `main.rs` refactored to remove ~1500 lines of historical discovery bloat.
    - Identification of obsolete research code in `crucible.rs`, `inference.rs`, and `probe.rs` for trimming.
    - Deprecation of systematic discovery tools (`sink.rs`) as focus shifts to model primitives.

### Functional Progress
- A working JIT harness (MAP_JIT, fork isolation, GPR snapshots)
- Proof that M4 uses ARM SME (not AMX)
- A 16×16 SGEMM kernel (PTRUE → ZERO ZA → [LD1W×2 + FMOPA + ADD×2]×K → [ST1W + ADD×2]×16)
- Fused GEMM + Bias + ReLU kernels (Gate 17)
- Differential correctness: `max_diff = 0.0` vs Accelerate
- Benchmark: 1.8–2.5× faster than Accelerate at tile-sized problems
- A 3-layer MNIST inference engine running entirely through JIT'd SME kernels (Gate 18)
- **1.93× faster than Accelerate** for full 3-layer MLP inference with pre-transposed input (Gate 20)
- Tiled GEMM up to 128×128 with branched K-loop, **5× faster at 16×16** (Gate 21)
- **Tiled inference engine**: 784→48→48→10 MLP using tiled GEMM, **1.13× vs Accelerate** (Gate 22)
- **Monolithic fused kernel**: single SMSTART/SMSTOP + vertical ST1W column-major stores, **1.55× vs Accelerate, 3.0 μs/batch** (Gate 23)

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

## Gate 20: Pre-transposed Input (Complete)

**Goal**: Eliminate the 784×16 input transpose — the single biggest bottleneck in Gate 19.

**Status**: Complete. **1.93× faster than Accelerate** for full 3-layer MLP inference.

**Approach**:
- Python exports `test_images_t.bin` — pre-transposed [784×16] column-major layout
- `MnistTestBatch` loads `images_t` field (falls back to runtime transpose if file missing)
- `CachedInferenceEngine::run_pretransposed()` skips the 784×16 transpose entirely
- Hidden-layer 16×16 transposes remain (only 256 floats each — negligible)

**Results**:

| Path | Latency | vs Accelerate |
|:-----|:--------|:-------------|
| Accelerate (cblas_sgemm) | 3.8 μs | 1.00× |
| Cached JIT (with transpose) | 9.1 μs | 0.42× |
| **Cached JIT (pretransposed)** | **2.0 μs** | **1.93×** |

The pre-transposed path saved **7.1 μs** — the transpose was 78% of total inference time.

**Performance journey** (Gates 18 → 20):

| Gate | Latency | vs Accelerate | Key optimization |
|:-----|:--------|:-------------|:-----------------|
| Gate 18 | 33.6 μs | 0.10× | Correctness proof |
| Gate 19 | 6.8 μs | 0.46× | Cached JIT pages |
| **Gate 20** | **2.0 μs** | **1.93×** | Pre-transposed input |

**Remaining optimization headroom**:
- Single SMSTART/SMSTOP across all 3 layers (saves ~0.5 μs from 4 mode switches)
- Column-major inter-layer stores (eliminates two 16×16 transposes, saves ~0.1 μs)
- These would push toward ~1.4 μs / ~2.7× vs Accelerate

## Gate 21: Tiled GEMM (Complete)

**Goal**: Break the 16×16 constraint — JIT-emit tiled SGEMM kernels for arbitrary M×N (multiples of 16, up to 128×128) with a branched K-loop.

**Status**: Complete. All sizes pass with **max_diff = 0.00e0** vs Accelerate.

**Architecture**:
- `build_sme_tiled_sgemm(m, n, k, act)` — emits inner kernel as `Vec<u32>`
- `build_sme_tiled_sgemm_page_cached(m, n, k, act, b_ptr, bias_ptr)` — builds callable JitPage
- Tile loop fully unrolled at JIT time (no runtime tile iteration overhead)
- K-loop uses SUBS/B.NE branch (7 instructions per tile, executes K times)
- Calling convention: X0=A (col-major), X1=C (row-major), X5=B (baked), X6=Bias (baked)
- Registers: X10/X11 base pointers, X4/X7 tile pointers, X8 K-counter, X9 scratch

**Benchmark (2000 samples, 500 warmup, median ns/call)**:

| Size | Tiles | JIT median | Accel median | Speedup |
|:-----|:------|:-----------|:-------------|:--------|
| 16×16×16 | 1 | 41 ns | 208 ns | **5.1×** |
| 32×32×32 | 4 | 83 ns | 250 ns | **3.0×** |
| 48×48×48 | 9 | 333 ns | 459 ns | **1.4×** |
| 64×64×64 | 16 | 834 ns | 667 ns | 0.8× |
| 128×128×128 | 64 | 7,750 ns | 3,000 ns | 0.4× |

> **Note**: An earlier benchmark without warmup showed 73× at 32×32 — this was a cold-start artifact.
> Accelerate's first call at each size pays a one-time ~6,600ns setup cost for internal buffer
> allocation and code path selection. After warmup, its dispatch tax is a flat **~200ns** regardless
> of matrix size.

**Key insight**: JIT dominates at ≤48×48 (zero dispatch overhead vs Accelerate's ~200ns fixed tax). Accelerate wins at ≥64×64 (optimized cache blocking, possible multi-core). The crossover at ~48×48 defines our niche: **tiny-model, latency-critical inference**.

**Not yet implemented**: Edge tiles (non-16-multiple dimensions), L2 cache blocking, double buffering.

## Gate 22: Tiled Inference Engine (Complete)

**Goal**: Replace the 16×16-only `CachedInferenceEngine` with the tiled GEMM infrastructure from Gate 21 — enable wider hidden layers and bigger models.

**Status**: Complete. 16/16 correct, `max_diff = 0.00e0`, **1.13× faster than Accelerate**.

**Architecture**: 784 → 48 (BiasReLU) → 48 (BiasReLU) → 10 (Bias, padded to 16)

Hidden dim 48 was chosen as the sweet spot — the largest dimension where tiled GEMM still beats Accelerate (1.4× at 48×48×48 per Gate 21 benchmarks). 48 = 3×16 tiles, so tiling logic gets a real workout.

**Components**:
- `scripts/train_mnist_wide.py` — trains 784→48→48→10 MLP, exports to `scripts/weights_wide/`
- `MnistWeightsWide` — parameterized weight loader with `config.txt` for hidden dim
- `TiledInferenceEngine` — uses `build_sme_tiled_sgemm_page_cached` for all layers
- `run_inference_reference_wide()` — Accelerate reference for differential testing

**Layer dimensions** (batch=16):

| Layer | M | N | K | Tiles | Activation |
|:------|:--|:--|:--|:------|:-----------|
| 1 | 16 | 48 | 784 | 1×3 | BiasReLU |
| 2 | 16 | 48 | 48 | 1×3 | BiasReLU |
| 3 | 16 | 16 | 48 | 1×1 | Bias |

**Results**:

| Metric | Value |
|:-------|:------|
| Predictions correct | 16/16 |
| Output max_diff | 0.00e0 |
| Build time (one-time) | 21.3 μs |
| Accelerate latency | 5.1 μs/batch |
| **Tiled JIT (pretransposed)** | **4.5 μs/batch** |
| Tiled JIT (with transpose) | 11.2 μs/batch |
| **vs Accelerate** | **1.13×** |

**Performance journey** (Gates 18 → 22):

| Gate | Architecture | Latency | vs Accelerate | Key optimization |
|:-----|:------------|:--------|:-------------|:-----------------|
| Gate 18 | 784→16→16→10 | 33.6 μs | 0.10× | Correctness proof |
| Gate 19 | 784→16→16→10 | 6.8 μs | 0.46× | Cached JIT pages |
| Gate 20 | 784→16→16→10 | 2.0 μs | 1.93× | Pre-transposed input |
| **Gate 22** | **784→48→48→10** | **4.5 μs** | **1.13×** | Tiled GEMM, wider model |

**Key insight**: Wider hidden layers (48 vs 16) push each layer's GEMM closer to the 48×48 crossover point where JIT and Accelerate are nearly matched. The JIT still wins overall due to zero dispatch overhead, but the margin narrows from 1.93× to 1.13×. This confirms the sweet spot: **models with hidden dims ≤48 benefit from JIT; larger models should use Accelerate**.

## Gate 23: Monolithic Fused Inference Kernel (Complete)

**Goal**: Emit all 3 layers into a single JitPage with one SMSTART/SMSTOP pair and zero inter-layer transposes — maximum kernel fusion.

**Status**: Complete. 16/16 correct, `max_diff = 0.00e0`, **1.55× faster than Accelerate**.

**Key innovations**:

1. **Single SMSTART/SMSTOP**: All 3 layers execute in one streaming session. Eliminates 4 redundant mode switches (~300-600 ns saved).

2. **ST1W Vertical Slices → Zero Transposes**: Intermediate layers use `ST1W ZA0V` (vertical) instead of `ST1W ZA0H` (horizontal). This stores ZA *columns* as 16 contiguous floats — which is exactly the column-major layout the next layer's LD1W expects. The transpose is eliminated entirely.

3. **LD1RW Broadcast Bias**: For column-major activation, `LD1RW` (load-and-replicate word) broadcasts a single bias float to all 16 Z register lanes. One bias element per column instead of loading the entire bias vector per row. Both `LD1RW` and `ST1W vertical` are confirmed available in streaming SVE mode on M4.

**Architecture**: `MonolithicInferenceEngine` — single JitPage, all pointers baked in.

**M4 streaming SVE discoveries**:
- `ST1W ZA0V` (vertical slices) **works correctly** in streaming mode ✓
- `LD1RW` (load-and-replicate word) **works correctly** in streaming mode ✓
- SVE gather loads (`LD1W scalar+vector`) are **NOT available** in streaming mode (hangs/faults)
- This confirms ARM's spec: gather/scatter operations are excluded from the streaming SVE subset

**Buffer strategy**:

| Layer | Input (A) | Output (C) | Store type |
|:------|:----------|:-----------|:-----------|
| Layer 1 | X0 (caller) | buf1 (col-major) | ST1W vertical |
| Layer 2 | buf1 | buf2 (col-major) | ST1W vertical |
| Layer 3 | buf2 | X1 (caller, row-major) | ST1W horizontal |

**Results**:

| Metric | Value |
|:-------|:------|
| Predictions correct | 16/16 |
| Output max_diff | 0.00e0 |
| Build time (one-time) | 17.8 μs |
| Accelerate latency | 4.6 μs/batch |
| Tiled JIT (Gate 22) | 4.6 μs/batch |
| **Monolithic JIT (Gate 23)** | **3.0 μs/batch** |
| **vs Accelerate** | **1.55×** |
| **vs Tiled (Gate 22)** | **1.55× speedup** |

**Performance journey** (Gates 18 → 23):

| Gate | Architecture | Latency | vs Accelerate | Key optimization |
|:-----|:------------|:--------|:-------------|:-----------------|
| Gate 18 | 784→16→16→10 | 33.6 μs | 0.10× | Correctness proof |
| Gate 19 | 784→16→16→10 | 6.8 μs | 0.46× | Cached JIT pages |
| Gate 20 | 784→16→16→10 | 2.0 μs | 1.93× | Pre-transposed input |
| Gate 22 | 784→48→48→10 | 4.5 μs | 1.13× | Tiled GEMM, wider model |
| **Gate 23** | **784→48→48→10** | **3.0 μs** | **1.55×** | Monolithic kernel, vertical ST1W |

**What the 1.55× speedup comes from**:
- ~1.6 μs saved from eliminating 3→1 SMSTART/SMSTOP (each pair ~500 ns)
- ~0.4 μs saved from eliminating 2 Rust transposes (16×48 = 768 scalar copies each)
- Zero Rust function call overhead between layers (no BLR/RET, no Rust stack frames)

## Gate 24: Clean Public API & Benchmarks (Complete)

**Goal**: Package the project for public consumption — clean API, proper error types, reproducible Criterion benchmarks, crate metadata.

**Status**: Complete.

**Deliverables**:

1. **`api.rs` — Public API surface**:
   - `SmeGemm` — build-once/call-many tiled SGEMM kernel. Owns weights/bias, validates dimensions, exposes `run()` and `run_row_major()`.
   - `SmeMlp` — fused multi-layer MLP. Owns all data, compiles into single JitPage. `run()` and `run_row_major()`.
   - `LayerConfig` — declarative layer specification (n, weights, bias, activation).
   - `SmeError` — proper error enum with `Display`/`Error` impls. No more `.unwrap()` in the public path.
   - `Activation` re-exported from `api` module (users don't need to import `emitter`).

2. **`lib.rs` — Public re-exports**:
   - `pub use api::{SmeGemm, SmeMlp, LayerConfig, Activation, SmeError}`
   - Internal modules remain `pub` for power users and benchmarks.
   - Module-level rustdoc with quick-start pointers.

3. **Criterion benchmarks expanded** — 5 groups:
   - `accelerate` — cblas_sgemm baseline at 16×16×K
   - `jit_cold` — fork-isolated kernel (measures safety harness overhead)
   - `jit_hot` — direct JitPage call (bare-metal throughput)
   - `fused` — GEMM+ReLU, GEMM+Bias+ReLU
   - `tiled` — **NEW**: SmeGemm API at 16×16, 32×32, 48×48, 64×64 vs Accelerate

4. **Cargo.toml metadata**:
   - Version bumped to 0.2.0
   - `license = "MIT OR Apache-2.0"`
   - `keywords`, `categories`, `readme` fields populated
   - Ready for crate distribution (not published — M4-only)

**Usage example** (SmeGemm):
```rust
use sme_jit_core::{SmeGemm, Activation};

let kernel = SmeGemm::new(16, 16, 32, &weights, None, Activation::None)?;
kernel.run(&input_col_major, &mut output);
```

**Usage example** (SmeMlp):
```rust
use sme_jit_core::{SmeMlp, LayerConfig, Activation};

let mut mlp = SmeMlp::new(784, &[
    LayerConfig { n: 48, weights: w1, bias: b1, activation: Activation::BiasReLU },
    LayerConfig { n: 48, weights: w2, bias: b2, activation: Activation::BiasReLU },
    LayerConfig { n: 16, weights: w3, bias: b3, activation: Activation::Bias },
])?;
mlp.run(&input_col_major, &mut output);
```

## Phase 2: Edge & Sequence Horizons (Gates 26–32)

With the foundational 16-multiple MLPs proven, the next phase targets arbitrary matrix sizes, multi-core scaling, and sequence model primitives (Transformers, RNNs, SSMs).

### Gate 26: Predicated Memory & Generation (Complete)
**Goal:** Emit SVE `WHILELT` instructions to generate dynamic predicate masks for edge bounds, and ensure `LD1W`/`ST1W` respect these masks.
**Status:** ✅ **Complete.**

**Results:**
- 20/20 elements copied correctly via predicated `LD1W` → `ST1W` loop.
- 12/12 guard elements at indices 20–31 remained untouched.
- `WHILELT` correctly generates lane masks for the 4-element tail (lanes 0–3 active on second iteration, lanes 4–15 inactive).

**Root cause of obstacle (resolved):** The `encode_sve_whilelt_s` encoder had two bugs:
1. **Wrong base**: `0x2590_0010` was missing bit 21 (fixed=1) and bit 12 (sf=64-bit), and incorrectly set bit 4 (eq=1). Correct base is `0x25a0_1400`.
2. **Wrong Rm shift**: Rm was placed at bits [15:11] (`<<11`) instead of the correct bits [20:16] (`<<16`).

The garbage encoding (`0x2591_107x`) decoded as undefined SVE, executed silently as a NOP on M4 — no SIGILL, no flag update, P0 remained zero, all LD1W/ST1W became no-ops under the all-false predicate. Diagnosed by cross-referencing against clang's authoritative disassembly. Pinned with a unit test covering three known reference values.

**M4 SVE discoveries:**
- `WHILELT Pd.S, Xn, Xm` (64-bit signed, .S) → `0x25a0_1400 | (Rm<<16) | (Rn<<5) | Pd` — confirmed via clang on M4.
- Undefined SVE encodings execute silently as NOPs on M4 (no SIGILL) — masking bugs that would fault-fast on stricter hardware.
- `LD1W`/`ST1W` with a fully-zero predicate are clean no-ops (no fault, no transfer).

### Gate 27: Predicated Outer Products (Complete)
**Goal:** Handle the K-loop tail by emitting `FMOPA` with properly masked predicate registers to zero out inactive MAC units.
**Status:** ✅ **Complete.**

**Results:**

| K | Label | Expected | Got | Diff | Cols masked |
|:--|:------|:---------|:----|:-----|:------------|
| 1 | trivial | −0.1000 | −0.1000 | 0.00e0 | ✓ |
| 7 | odd, prime | 7.2000 | 7.2000 | 0.00e0 | ✓ |
| 13 | prime | 62.6000 | 62.6000 | 0.00e0 | ✓ |
| 16 | full SVE width | 122.4000 | 122.4000 | 0.00e0 | ✓ |
| 31 | odd, prime | 934.0000 | 934.0000 | 0.00e0 | ✓ |
| 100 | larger | 32825.0078 | 32825.0078 | 0.00e0 | ✓ |

`ZA[0][1..15] = 0.0` for all K — confirming `FMOPA P1/M, P1/M` accumulates only into `ZA[0][0]`.

**Encoder bug fixed:**
`encode_sme_st1w_za_h` used `pg << 11` for the predicate field. Correct position is bits **12–10** (`pg << 10`), matching SVE LD1W/ST1W. Silent for P0 (zero in any position = 0), wrong for P1+. Pinned with `st1w_za_h_pg_field` unit test.

**`encode_sme_fmopa` added:**
Parametric encoder for `FMOPA ZAda.S, Pn/M, Pm/M, Zn.S, Zm.S`. Replaces four hard-coded `0x8081_0000` constants throughout the kernel builders. Pinned with `fmopa_encoding` unit test covering three reference values.

**M4 SME discoveries:**
- **SMSTART resets predicates to all-false.** Every kernel must emit `PTRUE P0.S` (and any other predicates in use) immediately after SMSTART. Kernels that rely on P0 for LD1W/ST1W without initialising it will silently produce all-zero output. Confirmed by gate27 initially outputting `c[0]=0.0` for all K; fixed by adding `PTRUE_P0_S` to prologue.
- **`FMOPA ZA0.S, Pn/M, Pm/M, Zn, Zm` with non-trivial predicates works correctly.** Only ZA entries where both row-predicate (Pn) and col-predicate (Pm) lanes are active get updated. ✓
- **`FMOPA P1/M, P1/M` modifies P1 as a side effect** after the first call on M4. ARM spec says predicates are read-only inputs to FMOPA — this is an undocumented M4 deviation. Workaround: re-run `WHILELT Pn` at the top of each FMOPA iteration to restore the mask.
- **Predicated ZA stores (`ST1W ZA0H, Pg≠P0`) behave unexpectedly** after ≥2 FMOPA iterations (writes more lanes than predicate specifies). Root cause unknown. Workaround: use P0 (all-true) for ZA extraction; mask output in the caller if needed.

### Gate 28: Arbitrary Tiled GEMM
**Goal:** Integrate Gates 26 and 27 into the main `SmeGemm` tiled architecture.
**Success:** `max_diff = 0.0` vs Accelerate for arbitrary M×N×K (e.g., 17×43×91) without physical memory padding.

### Gate 29: Multi-threading & P-Core Pinning
**Goal:** Dispatch large GEMMs across multiple P-cores to surpass Accelerate at ≥64×64.
**Success:** Multi-threaded JIT beats Accelerate at 128×128.

### Gate 30: Tiny-Transformer Primitives
**Goal:** Implement SVE-based Softmax approximation and LayerNorm/RMSNorm (horizontal reductions).
**Success:** Execute a single Self-Attention block ($Q K^T V$) natively inside the JIT.

### Gate 31: RNN / GEMV Specialized Kernel
**Goal:** Optimize a pure Matrix-Vector ($M \times 1$) kernel for sequence state updates without wasting ZA tiles.
**Success:** 10× speedup over framework dispatch for a batch-size=1 recurrent state update.

### Gate 32: SSM / Mamba Primitives
**Goal:** Emit 1D causal convolutions (`EXT` sliding windows) and hardware-aware associative parallel scans.
**Success:** JIT-compiled execution of a minimal Mamba block.

## Codebase Cleanup (Ongoing)

As the project matures, we are trimming research-phase artifacts to focus on performance:
- [x] Refactor `main.rs` into a lean research dispatcher (now consumes lib crate, zero unused-item warnings).
- [x] Trim `crucible.rs` — reduced to pure Accelerate FFI bindings (38 lines); `Crucible` struct and helper methods removed.
- [x] Delete `inference.rs` — all three engine types (`MonolithicInferenceEngine`, `TiledInferenceEngine`, reference path) had zero callers; `api.rs` uses `emitter` directly.
- [x] Trim `probe.rs` — systematic brute-force discovery removed; fork-isolation harness retained for benchmarks.
- [x] Remove `sink.rs` — JSONL sweep logger deleted entirely.
- [x] Trim `emitter.rs` — removed `build_sme_bfmopa_16x16`, `build_sme_smopa_16x16`, `build_sme_sgemm_page_cached`, and all dead encoders (BFMOPA/SMOPA/UMOPA/SUMOPA, MOVA, LDP_X). Net: −250 lines.
- [x] Delete `weights.rs` — only `inference.rs` consumed it; both removed together.

## Deferred (from original roadmap)

These items are deprioritized but not abandoned:

| Item | Original Gate | Status |
|:-----|:-------------|:-------|
| Multi-tile tiling (M,N > 16) | Gate 16 (old) | ✅ **Done** — Gate 21 |
| OS scheduler bypass (P-core pinning) | Gate 17 (old) | Deferred — nice for benchmarks, not critical path |
| Double-buffered loads | Gate 18 (old) | Deferred — optimization after new instruction discovery |
| Batched small-SGEMM | — | Deferred — build after fused kernels prove out |
| Single SM session across layers | — | ✅ **Done** — Gate 23 |
| Column-major inter-layer stores | — | ✅ **Done** — Gate 23 (vertical ST1W) |

## Non-Goals (Archived)

- ~~AMX instruction encoding~~ — dead on M4
- ~~Frida heist scripts~~ — exploration complete, data preserved in git history
- ~~Planning documents~~ — replaced by this roadmap
