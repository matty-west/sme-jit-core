#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod cpu_state;
mod crucible;
mod emitter;
mod jit_page;
mod leaf;
mod probe;
mod signal_handler;
mod sink;

use signal_handler::install_sigill_handler;



// --- Gate 14c: SME Pivot — FMOPA Smoke Test ---
//
// Root-cause analysis (senior_engineer_report.md):
//   • Frida Stalker traced 50K+ instructions in APL_sgemm but saw ZERO AMX
//     opcodes (0x0020_xxxx) in the execution stream.
//   • Synthetic AMX (LDX/LDY/FMA/STZ) and bit-walk both produce all-zeros.
//   • M4 Accelerate is using ARM standard SME (FMOPA), not Apple AMX.
//
// Strategy: prove the full SME pipeline works on M4:
//   SMSTART → PTRUE → DUP (load data) → FMOPA (compute) → ST1W (store) → SMSTOP
//
// Note: SMSTART/SMSTOP are emitted by the probe harness when streaming=true.
//       We only need the inner instructions.

// ═══════════════════════════════════════════════════════════════════════════════
// SME SGEMM Kernel Builder
// ═══════════════════════════════════════════════════════════════════════════════
//
// Generates a complete 16×16×K SGEMM as a sequence of SME opcodes.
//
// M4 SVL = 512 bits → 16 float32 per Z register → ZA0 is a 16×16 tile.
// One FMOPA computes 256 multiply-accumulates (full rank-1 outer product).
//
// Register ABI (set via overrides before SMSTART):
//   X2  = C output pointer  (SharedMemory, row-major 16×16)
//   X3  = 0                 (zero offset for LD1W / ST1W scalar+scalar)
//   X4  = A pointer         (column k at A + k*64 bytes — 16 contiguous floats)
//   X5  = B pointer         (row k    at B + k*64 bytes — 16 contiguous floats)
//   X12 = W12 = 0           (ST1W slice index, incremented during store)

/// SVE LD1W (scalar+scalar): `LD1W {Zt.S}, Pg/Z, [Xn, Xm, LSL #2]`
///
/// Encoding: `1010_0101_01_0_Rm[4:0]_010_Pg[2:0]_Rn[4:0]_Zt[4:0]`
/// Base (all fields zero): `0xA540_4000`
const fn encode_sve_ld1w_ss(zt: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    0xA540_4000
        | ((rm as u32) << 16)
        | ((pg as u32) << 10)
        | ((rn as u32) <<  5)
        | (zt as u32)
}

/// ADD Xd, Xn, #imm12 (64-bit immediate add, no shift)
const fn encode_add_x_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x9100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// ADD Wd, Wn, #imm12 (32-bit immediate add, no shift)
const fn encode_add_w_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x1100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// Build a complete SME SGEMM block for M=N=16 (one ZA tile), K iterations.
///
/// The block runs inside streaming mode (SMSTART/SMSTOP handled by the probe
/// harness). It assumes the register ABI documented above.
///
/// Layout:
///   PTRUE P0.S
///   ZERO {ZA}
///   [K × (LD1W Z0 ← A; LD1W Z1 ← B; FMOPA ZA0; ADD X4,#64; ADD X5,#64)]
///   [16 × (ST1W row; ADD W12,#1; ADD X2,#64)]
fn build_sme_sgemm_16x16(k: usize) -> Vec<u32> {
    use crate::emitter::{ZERO_ZA, encode_sme_st1w_za_h};

    const PTRUE_P0_S: u32 = 0x2598_E3E0;
    const SVL_BYTES: u16 = 64; // 512 bits = 64 bytes
    const TILE_ROWS: usize = 16;

    let ld1w_z0_x4 = encode_sve_ld1w_ss(0, 0, 4, 3); // LD1W Z0.S, P0/Z, [X4, X3, LSL#2]
    let ld1w_z1_x5 = encode_sve_ld1w_ss(1, 0, 5, 3); // LD1W Z1.S, P0/Z, [X5, X3, LSL#2]
    let fmopa_za0  = 0x8081_0000_u32;                  // FMOPA ZA0.S, P0/M, Z0.S, Z1.S
    let add_x4_svl = encode_add_x_imm(4, 4, SVL_BYTES);// ADD X4, X4, #64
    let add_x5_svl = encode_add_x_imm(5, 5, SVL_BYTES);// ADD X5, X5, #64
    let st1w_za0   = encode_sme_st1w_za_h(0, 0, 0, 2, 3); // ST1W {ZA0H.S[W12,#0]},P0,[X2,X3,LSL#2]
    let add_w12_1  = encode_add_w_imm(12, 12, 1);       // ADD W12, W12, #1
    let add_x2_svl = encode_add_x_imm(2, 2, SVL_BYTES); // ADD X2, X2, #64

    let mut block = Vec::with_capacity(2 + 5 * k + 3 * TILE_ROWS);

    // ── Setup ──
    block.push(PTRUE_P0_S);
    block.push(ZERO_ZA);

    // ── K-loop (unrolled) ──
    for _ in 0..k {
        block.push(ld1w_z0_x4);  // Z0 ← A column k
        block.push(ld1w_z1_x5);  // Z1 ← B row k
        block.push(fmopa_za0);    // ZA0 += Z0 ⊗ Z1
        block.push(add_x4_svl);   // A ptr += SVL bytes
        block.push(add_x5_svl);   // B ptr += SVL bytes
    }

    // ── Store all 16 rows of ZA0 ──
    for _ in 0..TILE_ROWS {
        block.push(st1w_za0);     // store row W12 → [X2]
        block.push(add_w12_1);    // W12++
        block.push(add_x2_svl);   // C ptr += SVL bytes
    }

    block
}


// ═══════════════════════════════════════════════════════════════════════════════
// Gate 14d: Full 16×16 SGEMM — Correctness vs Accelerate
// ═══════════════════════════════════════════════════════════════════════════════

fn gate_14d() {
    use crate::crucible::Crucible;
    use crate::probe::SharedMemory;

    println!("── gate 14d: full 16×16 SME SGEMM vs Accelerate ──");
    println!();

    let m = 16_usize;
    let n = 16_usize;
    let k = 16_usize;

    // ── Input matrices ──
    // A stored as K columns of M=16 floats (contiguous column layout for LD1W).
    // B stored as K rows of N=16 floats (contiguous row layout for LD1W).
    // Both all-ones → C[i][j] = K for all i,j.
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    // ── Output (SharedMemory for fork visibility) ──
    let c_shared = SharedMemory::<[f32; 256]>::new();
    let c_ptr = c_shared.as_mut_ptr() as *mut f32;
    // SAFETY: c_ptr points to SharedMemory of 256 f32s (1024 bytes).
    unsafe { std::ptr::write_bytes(c_ptr, 0, 256); }

    // ── Build the kernel ──
    let block = build_sme_sgemm_16x16(k);
    println!("  Kernel: {} instructions (K={}, M=N=16)", block.len(), k);
    println!("    Setup:   2 (PTRUE + ZERO_ZA)");
    println!("    Compute: {} ({}× [LD1W + LD1W + FMOPA + ADD + ADD])", 5 * k, k);
    println!("    Store:   {} (16× [ST1W + ADD W12 + ADD X2])", 3 * 16);
    println!();

    // ── Register overrides ──
    let overrides: Vec<(u8, u64)> = vec![
        (2,  c_ptr as u64),              // X2  = C output (SharedMemory)
        (3,  0_u64),                      // X3  = 0 (zero offset)
        (4,  a.as_ptr() as u64),         // X4  = A pointer
        (5,  b.as_ptr() as u64),         // X5  = B pointer
        (12, 0_u64),                      // X12 = W12 = 0 (slice index)
    ];

    // ── Execute ──
    let crucible = Crucible::new();
    println!("  Executing {}-instruction SME kernel...", block.len());
    let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

    println!();
    if result.faulted {
        println!("  [✗] FAULT: {}", result.status());
        println!("    Likely cause: LD1W encoding incorrect for streaming mode.");
        println!("    Fallback: try DUP-based kernel (no memory loads).");
        return;
    }

    // ── Read results ──
    // SAFETY: c_ptr points to SharedMemory, fork child wrote via ST1W.
    let c_jit: &[f32] = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
    let nonzero = c_jit.iter().filter(|&&v| v != 0.0).count();

    println!("  ✓ Kernel executed without fault!");
    println!("  c_jit[0..8]   = {:?}", &c_jit[0..8]);
    println!("  c_jit[248..256] = {:?}", &c_jit[248..256]);
    println!("  non-zero: {}/{}", nonzero, m * n);
    println!();

    // ── Compare vs Accelerate ──
    let c_accel = Crucible::run_accelerate(m, n, k, &a, &b);
    let max_diff = Crucible::max_abs_diff(&c_accel, c_jit);

    println!("  Accelerate c[0..8] = {:?}", &c_accel[0..8]);
    println!("  max_diff = {:.2e} (target < 1e-4)", max_diff);
    println!();

    if max_diff < 1e-4 {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🏆  GOLDEN — SME SGEMM MATCHES ACCELERATE ON M4  🏆   █");
        println!("  █                                                          █");
        println!("  █   max_diff = {:.2e}                                 █", max_diff);
        println!("  █   Tile: {}×{}, K={}                                    █", m, n, k);
        println!("  █   {} FMOPA ops × 256 MACs = {} total MACs           █", k, k * 256);
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else if nonzero == m * n {
        println!("  [!] All {} elements non-zero but diff too large.", m * n);
        println!("  c_jit[0] = {}, c_accel[0] = {}", c_jit[0], c_accel[0]);
        println!("  Check: K-loop count, or accumulator not fully zeroed.");
    } else {
        println!("  [!] Only {}/{} elements stored. ST1W loop may be incomplete.", nonzero, m * n);
    }

    println!();
    println!("✓ gate 14d complete\n");
}

fn main() {
    install_sigill_handler();
    gate_14d();
}
