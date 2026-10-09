# SRD-STIFF: Reproducible Benchmark Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)
[![OSF](https://img.shields.io/badge/OSF-registered-blue)](https://osf.io/XXXXX)
[![CI](https://github.com/srdlean/stiff-benchmarks/workflows/CI/badge.svg)](https://github.com/srdlean/stiff-benchmarks/actions)

**One-command reproducibility:**
```bash
docker run --rm srdlean/benchmarks:v1.0.0 \
  snakemake --cores 24 --use-singularity all
```

**Minimum deney (1 saat):**
```bash
docker run --rm srdlean/benchmarks:v1.0.0 \
  python scripts/run_all.py --category A --n-seeds 3
```

**Lean ispatları:** [github.com/srdlean/formalization](https://github.com/srdlean/formalization)

**Preregistration:** [osf.io/XXXXX](https://osf.io/XXXXX) (2025-07-15)

**Veri:** [zenodo.org/record/XXXXX](https://zenodo.org/record/XXXXX)

## Alıntı
```bibtex
@article{srd_stiff_2026,
  title={SRD-STIFF: A Formally Verified Spectral Regularization Method for Stiff Optimization},
  author={Dalman, H. and Badur Dalman, S.},
  journal={International Journal of Mathematical and Logical Computing},
  volume={17},
  pages={333},
  year={2026},
  doi={10.XXXX/XXXXX}
}
```
