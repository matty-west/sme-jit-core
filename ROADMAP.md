# sme-jit-core Roadmap

## Where We Are

Gates 0–15 are complete. We have:
- A working JIT harness (MAP_JIT, fork isolation, GPR snapshots)
- Proof that M4 uses ARM SME (not AMX)
- A 16×16 SGEMM kernel (PTRUE → ZERO ZA → [LD1W×2 + FMOPA + ADD×2]×K → [ST1W + ADD×2]×16)
- Differential correctness: `max_diff = 0.0` vs Accelerate
- Benchmark: 1.8–2.5× faster than Accelerate at tile-sized problems

## Gate 16: BFMOPA / SMOPA Probing

**Goal**: Discover which extended outer product instructions M4 supports.

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

## Gate 17: Fused GEMM + Activation Kernels

**Goal**: JIT-emit kernels that fuse matmul with activation functions in a single kernel — zero intermediate memory traffic.

**Fusion targets** (in order of complexity):

1. **GEMM + ReLU**: After FMOPA accumulation, apply `FMAX Zn.S, Pn/M, Zn.S, #0.0` before ST1W. This is a single SVE instruction — trivial to add.

2. **GEMM + Bias**: Load a bias vector into a Z register, add it to each output row before store. Requires one extra LD1W + FADD per row.

3. **GEMM + GELU**: Approximate GELU via `x * sigmoid(1.702 * x)` using SVE FMUL/FADD/FRECPE. More instructions but still fused — no memory round-trip.

4. **GEMM + Bias + ReLU**: Chain bias add and activation.

**Build plan**:
1. Extend `build_sme_sgemm_16x16` to accept an `Activation` enum parameter
2. Emit activation instructions between FMOPA loop and ST1W store loop
3. Build correctness tests: compute reference in Rust (matmul + activation), compare against JIT output
4. Benchmark fused kernel vs separate Accelerate SGEMM + vDSP activation calls

**Stretch goal**: Chain 2-3 fused layers into a tiny MNIST inference engine. One binary, zero frameworks, classification in <1μs.

## Gate 18: Tiny Inference Engine Demo

**Goal**: Run a small neural network (2-3 layer MLP) entirely through JIT'd fused SME kernels.

- Pre-train a tiny MNIST classifier in Python, export weights as raw f32 arrays
- At startup, JIT-compile one fused kernel per layer (GEMM + bias + activation)
- Chain kernel calls: input → layer1 → layer2 → output
- Benchmark end-to-end latency vs CoreML / MLX / Accelerate
- Measure the "framework tax" — how much overhead do ML frameworks add for tiny models?

## Deferred (from original roadmap)

These items are deprioritized but not abandoned:

| Item | Original Gate | Status |
|:-----|:-------------|:-------|
| Multi-tile tiling (M,N > 16) | Gate 16 (old) | Deferred — focus on depth before breadth |
| OS scheduler bypass (P-core pinning) | Gate 17 (old) | Deferred — nice for benchmarks, not critical path |
| Double-buffered loads | Gate 18 (old) | Deferred — optimization after new instruction discovery |
| Batched small-SGEMM | — | Deferred — build after fused kernels prove out |

## Non-Goals (Archived)

- ~~AMX instruction encoding~~ — dead on M4
- ~~Frida heist scripts~~ — exploration complete, data preserved in git history
- ~~Planning documents~~ — replaced by this roadmap
