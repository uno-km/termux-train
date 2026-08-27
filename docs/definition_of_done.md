# Definition of Done (DoD) & Release Readiness Checklist

> **Framework:** `termux-train`  
> **Baseline:** Production Grade Edge Training Verification

---

## 1. Automated Test Criteria
- 100% of unit tests in `tests/` must pass without regressions.
- Numerical gradient verification (`gradcheck`) tolerance strictly within `eps <= 1e-4`.
- Memory stability: Zero resident set size (RSS) memory creep over 50 consecutive training epochs.

## 2. On-Device Verification (Physical Galaxy Devices)
- Successful execution of Tiny Transformer forward and backward passes on ARM64 Termux.
- Verification of FP16 and FP32 numerical convergence on real silicon.
- Safe termination handling on OS memory pressure signals.

## 3. Documentation & Packaging
- Clean, 100% English documentation across all guides and docstrings.
- OpenSSF and Apache-2.0 license compliance verification.
