# Validation

Checks performed on 2026-10-24.

| Record | Scope |
|---|---|
| smoke_cpu.json | Synthetic 64×64 input; forward, backward, and inference |
| smoke_real_512.json | One real 512×512 image with COCO initialization and CLIP text features |
| clip_parity.json | Text-encoder comparison with official CLIP on 12 descriptions |
| workflow_checks.json | Two batch-8 optimizer updates, checkpoint resume, two-image evaluation, and CPU prediction |
| portable_cpu.json | CPU checks from the extracted archive |
| package_audit.json | Package syntax and file checks |

Run `python tests/test_contracts.py` for cache, sampler, and evaluation tests, or `python smoke_test.py --device cpu` for model checks.
