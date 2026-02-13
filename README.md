# QuantumHallVMC

Variational Monte Carlo code used in "Quantum melting Wigner crystals into Hall liquids" by Aidan P. Reddy and Liang Fu.

The code implements VMC optimization and sampling of a Slater-Jastrow wavefunction for spinless electrons on a 2D torus, both with and without magnetic field. Supported orbital constructions include:

- Landau-level magnetic Bloch states
- Plane waves
- Gaussian orbitals with magnetic or periodic boundary conditions

Wavefunction optimization uses stochastic reconfiguration.

See supplemental_material.pdf for further discussion of methods.

## Quickstart

From the repository root:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuantumHallVMC; println("QuantumHallVMC loaded")'
```

## Examples

- `notebooks/demo.ipynb`: Landau-level / plane-wave orbital workflow
- `notebooks/demo_gaussian.ipynb`: Gaussian orbital workflow

## Installation and dependencies

See `INSTALL.md`.

## Citation

If you use this code, please cite the associated paper. See `CITATION.cff`.

```bibtex
@article{reddy2025quantum,
  title={Quantum melting a Wigner crystal into Hall liquids},
  author={Reddy, Aidan P and Fu, Liang},
  journal={arXiv preprint arXiv:2508.21000},
  year={2025}
}
```
