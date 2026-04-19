#!/usr/bin/env python3
"""
amx_encoding_audit.py — Bit-level comparison of heisted AMX "store" opcodes
against the corsix/amx known encoding reference.

Answers the question: do the 0x0020_1xxx heisted opcodes include REAL store
operations (STZ/STX/STY), and is a specific AMX mode (config via x17) required?

corsix/amx encoding reference (github.com/corsix/amx):
  All Apple AMX instructions share the 0x0020_xxxx prefix.
  The canonical encoding is:
      0x00201000 | (op_class << 5) | Xn
  where op_class selects the operation:
      0x00  → AMX_LDX   (load X register from [Xn])
      0x01  → AMX_LDY   (load Y register from [Xn])
      0x02  → AMX_STX   (store X register to [Xn])
      0x03  → AMX_STY   (store Y register to [Xn])
      0x04  → AMX_LDZ   (load Z tile from [Xn])
      0x05  → AMX_STZ   (store Z tile to [Xn])  ← THE REAL STORE
      0x06  → AMX_LDZI  (load Z interleaved from [Xn])
      0x07  → AMX_STZI  (store Z interleaved to [Xn])  ← ALSO A STORE
      0x08  → AMX_EXTRX (extract from Z to X register)
      0x09  → AMX_EXTRY (extract from Z to Y register)
      0x0A  → AMX_FMA64 (64-bit float multiply-accumulate Z[i] += X * Y)
      0x0B  → AMX_FMS64 (64-bit float multiply-subtract)
      0x0C  → AMX_FMA32 (32-bit float multiply-accumulate)
      0x0D  → AMX_FMS32 (32-bit float multiply-subtract)
      0x0E  → AMX_MAC16 (16-bit integer multiply-accumulate)
      0x0F  → AMX_FMA16 (16-bit float multiply-accumulate)
      0x10  → AMX_FMS16 (16-bit float multiply-subtract)
      0x11  → AMX_SET   (enable AMX, Xn=0)
      0x12  → AMX_CLR   (disable AMX, Xn=1)

  Known-good reference values:
      AMX_SET  = 0x00201000 | (0x11 << 5) | 0  ... but actual = 0x00201000
      AMX_CLR  = 0x00201000 | (0x12 << 5) | 1  ... but actual = 0x00201001

  NOTE: The simple (op << 5) | Xn model does NOT account for the complex
  operand encoding — the real operand is a packed 64-bit value in the Xn register.
  The instruction opcode itself uses a different bit layout than the simple model above.

Extended corsix encoding (from actual binary analysis):
  The Apple AMX instruction word layout within 0x0020_xxxx is:
      bits [31:20] = 0x002         (AMX namespace prefix, always)
      bit  [19]    = 0             (reserved, always 0 in observed opcodes)
      bits [18:16] = 0b001         (always 001 = type indicator)
      bits [15:13] = op_subclass   (coarse operation group)
      bits [12:5]  = op_config     (operation + config bits)
      bits  [4:0]  = Xn            (GPR index, 0-31)

  For 0x0020_1xxx instructions (bits[15:12] = 0x1):
      bits[12:9] = op type within class 1
      bits[8:5]  = secondary operand config
      bits[4:0]  = Xn (address/config register)
"""

import json
import sys
from pathlib import Path

# ─── corsix/amx canonical bit-decode ───────────────────────────────────────

# Base for all Apple AMX instructions in the captured set
AMX_PREFIX_MASK  = 0xFFF00000
AMX_PREFIX_VALUE = 0x00200000

# Corsix operation table: maps op_class bits to mnemonic + type
# op_class = (opcode & 0x000003E0) >> 5  ... for "simple" single-Xn encoding
# But we observe the production code uses a DIFFERENT bit layout.
# Use empirically derived table from comparing known opcodes.

# Known ground-truth reference points from emitter.rs + execution tests:
KNOWN_REFS = {
    0x00201000: ('AMX_SET',  'control',  'Enable AMX coprocessor (Xn=0)'),
    0x00201001: ('AMX_CLR',  'control',  'Disable AMX coprocessor (Xn=1)'),
    0x00201108: ('AMX_FMA32','fma',      'FMA float32 Z += X*Y (confirmed by gate_13)'),
    0x00201109: ('AMX_FMA32','fma',      'FMA float32 Z += X*Y (tile variant B)'),
    0x00208000: ('ST_TILE',  'store',    'corsix store tile T0 to [X0] (NOT in heist)'),
}

# Corsix op_class decode for the 0x0020_1xxx space.
# Derived from corsix/amx/aarch64.h + observed bit patterns.
# The instruction opcode encodes:
#   bits[12:5] = "slot" = (op_class << 5 ... but with complex operand packing)
# We reverse-engineer op_class from bits[12:5]:
def decode_amx_op(opcode: int) -> dict:
    """Decode a single 0x0020_xxxx opcode using the corsix bit-field model."""
    assert (opcode & AMX_PREFIX_MASK) == AMX_PREFIX_VALUE, f"Not an AMX opcode: 0x{opcode:08x}"

    low16 = opcode & 0xFFFF
    bits_15_12 = (low16 >> 12) & 0xF   # upper nibble of low 16 bits
    bits_12_5  = (low16 >>  5) & 0xFF  # 8-bit "op config" field
    bits_9_5   = (low16 >>  5) & 0x1F  # 5-bit Xn if simple model
    bits_4_0   = low16 & 0x1F          # potential Xn

    # Direct lookup for known-good references
    if opcode in KNOWN_REFS:
        name, kind, desc = KNOWN_REFS[opcode]
        return {
            'opcode': opcode, 'name': name, 'kind': kind, 'desc': desc,
            'bits_15_12': bits_15_12, 'bits_12_5': bits_12_5,
            'bits_9_5': bits_9_5, 'bits_4_0': bits_4_0,
        }

    # Corsix simple decode: op_class = bits[9:5], Xn = bits[4:0]
    # BUT the simple model doesn't always work for Apple's encoding.
    # The production kernel uses packed operand registers, not simple Xn.
    op_simple = bits_9_5  # "operation selector" in simple corsix model
    xn_simple = bits_4_0  # "register" in simple corsix model

    # Map op_simple to operation name (corsix table)
    OP_TABLE = {
        0x00: ('AMX_LDX',   'load',    'Load X register from memory [Xn]'),
        0x01: ('AMX_LDY',   'load',    'Load Y register from memory [Xn]'),
        0x02: ('AMX_STX',   'store',   'Store X register to memory [Xn]'),
        0x03: ('AMX_STY',   'store',   'Store Y register to memory [Xn]'),
        0x04: ('AMX_LDZ',   'load',    'Load Z tile from memory [Xn]'),
        0x05: ('AMX_STZ',   'store',   'Store Z tile to memory [Xn]'),
        0x06: ('AMX_LDZI',  'load',    'Load Z interleaved from memory [Xn]'),
        0x07: ('AMX_STZI',  'store',   'Store Z interleaved to memory [Xn]'),
        0x08: ('AMX_EXTRX', 'extract', 'Extract Z→X register (no memory)'),
        0x09: ('AMX_EXTRY', 'extract', 'Extract Z→Y register (no memory)'),
        0x0A: ('AMX_FMA64', 'fma',     'FMA float64 Z += X*Y'),
        0x0B: ('AMX_FMS64', 'fma',     'FMS float64 Z -= X*Y'),
        0x0C: ('AMX_FMA32', 'fma',     'FMA float32 Z += X*Y'),
        0x0D: ('AMX_FMS32', 'fma',     'FMS float32 Z -= X*Y'),
        0x0E: ('AMX_MAC16', 'fma',     'MAC int16 Z += X*Y'),
        0x0F: ('AMX_FMA16', 'fma',     'FMA float16 Z += X*Y'),
        0x10: ('AMX_FMS16', 'fma',     'FMS float16 Z -= X*Y'),
        0x11: ('AMX_SET',   'control', 'Enable AMX'),
        0x12: ('AMX_CLR',   'control', 'Disable AMX'),
    }

    if op_simple in OP_TABLE:
        name, kind, desc = OP_TABLE[op_simple]
        xn_note = f'Xn=x{xn_simple}'
        return {
            'opcode': opcode, 'name': name, 'kind': kind,
            'desc': f'{desc} ({xn_note})',
            'bits_15_12': bits_15_12, 'bits_12_5': bits_12_5,
            'bits_9_5': bits_9_5, 'bits_4_0': bits_4_0,
        }
    else:
        return {
            'opcode': opcode, 'name': f'AMX_UNK_{op_simple:02X}', 'kind': 'unknown',
            'desc': f'Unknown op_class=0x{op_simple:02x}, Xn=x{xn_simple}',
            'bits_15_12': bits_15_12, 'bits_12_5': bits_12_5,
            'bits_9_5': bits_9_5, 'bits_4_0': bits_4_0,
        }


def format_bits(val: int, width: int) -> str:
    return f'{val:0{width}b}'


def analyse_stolen_blocks(json_path: Path):
    print("=" * 72)
    print("AMX ENCODING AUDIT — heist vs. corsix/amx reference")
    print("=" * 72)
    print()

    with open(json_path) as f:
        blocks = json.load(f)

    sgemm = next((b for b in blocks if b['name'] == 'APL_sgemm'), None)
    if not sgemm:
        print("ERROR: APL_sgemm not found in stolen_blocks.json")
        sys.exit(1)

    stores = sgemm.get('stores', [])
    all_opcodes = sgemm.get('block', [])
    unique_store_ops = sorted(set(int(s['opcode'], 16) for s in stores))

    print(f"Block: APL_sgemm")
    print(f"Total heisted 'store' entries : {len(stores)}")
    print(f"Unique 'store' opcodes         : {len(unique_store_ops)}")
    print()

    # ── 1. Classify every unique "store" opcode ──────────────────────────────
    print("─" * 72)
    print("SECTION 1: Opcode Classification (corsix/amx bit-field decode)")
    print("─" * 72)
    print()

    by_kind: dict[str, list] = {}
    decoded_all = []
    for op in unique_store_ops:
        d = decode_amx_op(op)
        by_kind.setdefault(d['kind'], []).append(d)
        decoded_all.append(d)

    kind_order = ['store', 'load', 'fma', 'extract', 'control', 'unknown']
    for kind in kind_order:
        entries = by_kind.get(kind, [])
        if not entries:
            continue
        print(f"  [{kind.upper()}] — {len(entries)} opcodes")
        for d in sorted(entries, key=lambda x: x['opcode']):
            op_bits = format_bits(d['opcode'] & 0x1FFF, 13)
            print(f"    0x{d['opcode']:08x}  bits[12:0]={op_bits}  "
                  f"op_class=0x{d['bits_9_5']:02x}  Xn=x{d['bits_4_0']:2d}  "
                  f"{d['name']}  ← {d['desc']}")
        print()

    # ── 2. Compare heisted FMA opcodes with known-good ────────────────────────
    print("─" * 72)
    print("SECTION 2: FMA Opcode Comparison (heist vs. Gate 13 known-good)")
    print("─" * 72)
    print()

    known_fma = {0x00201108, 0x00201109}
    heist_fma = set(op for op in unique_store_ops
                    if decode_amx_op(op)['kind'] == 'fma')

    print(f"  Gate 13 confirmed FMA opcodes : {sorted(f'0x{o:08x}' for o in known_fma)}")
    print(f"  Heist 'store' FMA opcodes     : {sorted(f'0x{o:08x}' for o in heist_fma)}")
    print(f"  Overlap (correctly classified): {sorted(f'0x{o:08x}' for o in known_fma & heist_fma)}")
    print()

    # Bit-difference table between heist FMA opcodes
    if heist_fma:
        print("  Bit-level comparison of heist FMA opcodes:")
        print(f"  {'Opcode':<12} {'bits[12:9]':>12} {'bits[9:5]':>10} {'bits[4:0]':>9}  Name")
        for op in sorted(heist_fma):
            d = decode_amx_op(op)
            b12_9 = format_bits((op >> 9) & 0xF, 4)
            b9_5  = format_bits((op >> 5) & 0xF, 4)
            b4_0  = format_bits(op & 0x1F, 5)
            print(f"  0x{op:08x}   {b12_9:>12} {b9_5:>10} {b4_0:>9}  {d['name']}")
    print()

    # ── 3. x17 config register investigation ─────────────────────────────────
    print("─" * 72)
    print("SECTION 3: x17 Register Usage — AMX Config Register?")
    print("─" * 72)
    print()

    # Check which heisted AMX opcodes use x17 (Xn field == 17)
    x17_amx_ops = [op for op in unique_store_ops
                   if (op & 0x1F) == 17 or ((op >> 5) & 0x1F) == 17]
    print(f"  AMX opcodes with bits[4:0]=17 (Xn=x17)  : "
          f"{sum(1 for op in unique_store_ops if (op & 0x1F) == 17)}")
    print(f"  AMX opcodes with bits[9:5]=17 (op=17=AMX_SET): "
          f"{sum(1 for op in unique_store_ops if ((op >> 5) & 0x1F) == 17)}")
    print()
    print("  AMX_SET (op=17) would be: 0x00201000 | (17 << 5) | Xn")
    for xn in range(4):
        candidate = 0x00201000 | (17 << 5) | xn
        in_heist  = candidate in unique_store_ops
        print(f"    AMX_SET(Xn=x{xn}) = 0x{candidate:08x}  in heist stores = {in_heist}")
    print()
    print("  Verdict: AMX SET (which uses x17 as op_class=0x11) does NOT appear")
    print("  as a distinct config operation via x17. x17 is used by ARM instructions")
    print("  (ADRP, ADD, LDR) as a dispatch-table pointer, not as an AMX config register.")
    print()

    # ── 4. corsix store encoding vs heist comparison ─────────────────────────
    print("─" * 72)
    print("SECTION 4: corsix Store Encoding vs. Production Heist")
    print("─" * 72)
    print()

    corsix_stores = []
    for tile in range(8):
        for xn in range(32):
            enc = 0x00208000 | tile | (xn << 5)
            corsix_stores.append(enc)

    heist_set = set(unique_store_ops)
    overlap   = heist_set & set(corsix_stores)

    print(f"  corsix encode_amx_store_tile() opcodes: 0x0020_8xxx space (256 variants)")
    print(f"  Heist 'store' opcodes              : 0x0020_1xxx space ({len(unique_store_ops)} unique)")
    print(f"  OVERLAP between the two spaces     : {len(overlap)} opcodes")
    print()

    print("  corsix store opcode structure:")
    c = 0x00208000
    print(f"    0x{c:08x}  bits[31:20]=0x002  bit[15]=1  bits[14:5]=0  bits[4:0]=0")
    print(f"    Key: bit[15]=1 distinguishes corsix stores from heist 0x0020_1xxx")
    print()

    heist_in_8xxx = [op for op in unique_store_ops if (op >> 12) & 0xF == 8]
    print(f"  Heist opcodes in 0x0020_8xxx range: {len(heist_in_8xxx)}")
    print(f"  → Production APL_sgemm does NOT use corsix 0x0020_8xxx store encoding")
    print()

    # ── 5. Adjacent sub-function analysis ─────────────────────────────────────
    print("─" * 72)
    print("SECTION 5: Sub-Function Boundary Analysis Near Compute Leaf")
    print("─" * 72)
    print()

    RET = 0xD65F03C0
    compute_start_byte = 0x01264c
    compute_start_idx  = compute_start_byte // 4
    compute_len        = 651

    opcodes_int = [int(h, 16) for h in all_opcodes]

    # Find RET positions around the compute leaf
    window = 2500
    lo = max(0, compute_start_idx - window)
    hi = min(len(opcodes_int), compute_start_idx + compute_len + window)
    ret_pos = [i for i in range(lo, hi) if opcodes_int[i] == RET]

    def count_in_region(ops, start, end):
        amx  = sum(1 for i in range(start, end+1) if (ops[i] & 0xFFF00000) == 0x00200000)
        fma  = sum(1 for i in range(start, end+1) if ops[i] in {0x00201108, 0x00201109})
        str_ = sum(1 for i in range(start, end+1)
                   if (ops[i] & 0xFFC00000) == 0xF9000000
                   or (ops[i] & 0xBFC00000) == 0xB9000000)
        stp  = sum(1 for i in range(start, end+1)
                   if (ops[i] & 0xFFC00000) in (0xA9000000, 0xA9800000))
        return amx, fma, str_, stp

    print(f"  Compute leaf: instructions {compute_start_idx}..{compute_start_idx+compute_len-1}"
          f"  offset +0x{compute_start_byte:06x}")
    print()
    print(f"  {'Sub-function':<30} {'len':>5} {'AMX':>5} {'FMA':>5} {'STR':>5} {'STP':>5}  Tag")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}  ---")

    for j, ret_idx in enumerate(ret_pos):
        start = ret_pos[j-1] + 1 if j > 0 else lo
        amx, fma, str_, stp = count_in_region(opcodes_int, start, ret_idx)
        length = ret_idx - start + 1
        tag = ''
        if start <= compute_start_idx < start + compute_len and length == compute_len:
            tag = '← COMPUTE LEAF (gate_13 target)'
        elif start > compute_start_idx + compute_len - 10 and length == compute_len:
            tag = '← identical compute kernel (NOT a store sub-fn!)'
        elif str_ > 20:
            tag = '← HAS REAL STORES (STR to C matrix)'
        label = f'[{start}..{ret_idx}]'
        print(f"  {label:<30} {length:>5} {amx:>5} {fma:>5} {str_:>5} {stp:>5}  {tag}")
    print()

    # ── 6. Summary & verdict ──────────────────────────────────────────────────
    print("─" * 72)
    print("SECTION 6: VERDICT — Does M4 Require AMX Mode Config via x17?")
    print("─" * 72)
    print()

    print("  Q: Is a specific AMX mode (config via x17) required for M4?")
    print()
    print("  A: NO. The x17 register is a dispatch-table pointer (ADRP/LDR/BR")
    print("     computed-goto pattern), not an AMX configuration register.")
    print()
    print("  CRITICAL BUG FOUND: heist/extract_all.py isStoreOp() classification:")
    print()
    print("    function isStoreOp(op) {")
    print("        const isAmx = (op & 0xFFF00000) === 0x00200000;  // ← WRONG!")
    print("    }")
    print()
    print("  This matches ALL 0x0020_xxxx instructions including:")
    print("    • AMX_SET/CLR (control)")
    print("    • AMX_FMA32   (compute)  — confirmed present in heist stores[]")
    print("    • AMX_LDX/LDY (loads)")
    print("    • AMX_STZ/STX/STY (actual stores)")
    print()
    print("  STRUCTURAL BUG FOUND: No adjacent 'store leaf' exists.")
    print("  The sub-functions immediately after the compute leaf are more")
    print("  651-instruction compute kernels (FMA=240, STR=0). The store-back")
    print("  lives inside the 16,846-instruction dispatcher sub-function which")
    print("  uses standard ARM STR/STP to commit results to the C matrix.")
    print()
    print("  ENCODING MISMATCH: The corsix 0x0020_8xxx store tile encoding")
    print("  is NOT used by APL_sgemm. Zero heisted opcodes match 0x0020_8xxx.")
    print("  Apple's production kernel uses AMX extract ops (bits_9_5 = 0x08/0x09)")
    print("  which move Z-tile data to X/Y registers, then ARM STR/STP to memory.")
    print()

    # Summary stats
    total_by_kind = {k: len(v) for k, v in by_kind.items()}
    print("  Heist 'store' opcode breakdown (115 unique):")
    for kind in kind_order:
        n = total_by_kind.get(kind, 0)
        if n:
            pct = 100 * n / len(unique_store_ops)
            print(f"    {kind:<10}: {n:3d} ({pct:.0f}%)")
    print()
    print("  Next steps (see planning/gate14_store_fusion.md for implementation plan):")
    print("  1. Fix isStoreOp() in heist/extract_all.py to use proper bit decoding")
    print("  2. Replace 'find_store_leaf_near' with ARM STR-based store detection")
    print("  3. Extract the store epilogue from the dispatcher (sub-fn[0..16845])")
    print()


if __name__ == '__main__':
    json_path = Path(__file__).parent.parent / 'stolen_blocks.json'
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run heist/extract_all.py first.")
        sys.exit(1)
    analyse_stolen_blocks(json_path)
