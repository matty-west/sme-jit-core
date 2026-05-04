//! Gate runner — tiny dispatcher for active research gates.
//!
//! All shared infrastructure lives in the library crate (`sme_jit_core`).
//! Historical gates have been retired; only the current research front
//! (Gate 26: predicated memory) is wired up here.

use sme_jit_core::emitter::{build_gate26_page, build_gate27_page};
use sme_jit_core::signal_handler::install_sigill_handler;

/// Current active research gate.
fn gate_26() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Gate 26: Predicated Memory & Generation — Edge Bounds");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let limit = 20_usize;
    let guard_val = -1.0f32;

    let src: Vec<f32> = (0..limit).map(|i| i as f32).collect();
    let mut dst = vec![guard_val; 32];

    println!("  [1] Building predicated copy kernel (limit={})...", limit);
    let page = build_gate26_page(limit).expect("Failed to build gate 26 page");

    println!("  [2] Executing copy...");
    // SAFETY: page contains valid SVE kernel, src and dst pointers are valid.
    unsafe {
        page.call_with_args(src.as_ptr() as u64, dst.as_mut_ptr() as u64);
    }

    println!("  [3] Verifying results...");
    let mut errors = 0;
    for i in 0..limit {
        if dst[i] != src[i] {
            println!("      [✗] Mismatch at index {}: expected {}, got {}", i, src[i], dst[i]);
            errors += 1;
        }
    }

    let mut guard_violations = 0;
    for i in limit..32 {
        if dst[i] != guard_val {
            println!("      [✗] Guard violation at index {}: expected {}, got {}", i, guard_val, dst[i]);
            guard_violations += 1;
        }
    }

    if errors == 0 && guard_violations == 0 {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🛡️  GATE 26 — PREDICATED MEMORY SUCCESS  🛡️           █");
        println!("  █                                                          █");
        println!("  █   Copied: {}/20 elements correctly                    █", limit);
        println!("  █   Guard:  12/12 elements untouched                       █");
        println!("  █                                                          █");
        println!("  █   SVE WHILELT generated correct masks for 20 elements.   █");
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else {
        println!("  [!] Gate 26 FAILED: {} errors, {} guard violations", errors, guard_violations);
    }

    println!();
    println!("✓ gate 26 complete\n");
}

/// Current active research gate.
fn gate_27() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Gate 27: Predicated Outer Products — Odd-K Dot Products");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let cases: &[(usize, &str)] = &[
        (1,   "trivial"),
        (7,   "odd, prime"),
        (13,  "prime"),
        (16,  "full SVE width"),
        (31,  "odd, prime"),
        (100, "larger"),
    ];

    let mut all_ok = true;
    for &(k, label) in cases {
        // Deterministic inputs: a[i] = i*0.1 - 0.1, b[i] = i+1.0
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 0.1).collect();
        let b: Vec<f32> = (0..k).map(|i| i as f32 + 1.0).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let mut c = vec![0.0f32; 16];
        let page = build_gate27_page(
            k,
            a.as_ptr() as u64,
            b.as_ptr() as u64,
            c.as_mut_ptr() as u64,
        ).expect("build_gate27_page failed");

        // SAFETY: all pointers baked in; page ends with RET.
        unsafe { page.call_void(); }

        let diff = (c[0] - expected).abs();
        let cols_masked = c[1..16].iter().all(|&x| x == 0.0);
        let pass = diff <= 1.6e-5 && cols_masked;
        if !pass { all_ok = false; }

        println!(
            "  K={:<4} ({:<14}) │ expected={:>10.4}  c[0]={:>10.4}  diff={:.2e}  cols_masked={}  {}",
            k, label, expected, c[0], diff,
            if cols_masked { "✓" } else { "✗" },
            if pass { "✓" } else { "✗" },
        );
    }

    println!();
    if all_ok {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🛡️  GATE 27 — PREDICATED OUTER PRODUCTS SUCCESS  🛡️   █");
        println!("  █                                                          █");
        println!("  █   6/6 test cases pass (K=1,7,13,16,31,100)              █");
        println!("  █   ZA[0][1..15] = 0.0 for all K — FMOPA P1/M verified   █");
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else {
        println!("  [!] Gate 27 FAILED — see above for mismatches");
    }
    println!();
    println!("✓ gate 27 complete\n");
}

fn main() {
    install_sigill_handler();

    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"gate27".to_string()) {
        gate_27();
    } else if args.contains(&"gate26".to_string()) {
        gate_26();
    } else if args.contains(&"all".to_string()) {
        println!("Historical gates are disabled by default. Run specifically (e.g., cargo run -- gate27).");
    } else {
        println!("sme-jit-core gate runner");
        println!("Usage: cargo run --release -- [gate27|gate26]");
        println!();
        println!("Running latest research (Gate 27)...");
        gate_27();
    }
}
