# QuantumHallVMC

Variational Monte Carlo code used in "Quantum melting Wigner crystals into Hall liquids" by Aidan P. Reddy and Liang Fu.

The code implements VMC optimization and sampling of a Slater-Jastrow wavefunction for spinless electrons on a 2D torus, both with and without magnetic field. Supported orbital constructions include:

- Landau-level magnetic Bloch states
- Plane waves
- Gaussian orbitals with magnetic or periodic boundary conditions

Wavefunction optimization uses stochastic reconfiguration.

## Quickstart

From the repository root:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuantumHallVMC; println("QuantumHallVMC loaded")'
```

Run tests:

```bash
julia --project=. -e 'import Pkg; Pkg.test()'
```

## Examples

- `/Users/aidanreddy/Desktop/VMC/notebooks/demo.ipynb`: Landau-level / plane-wave orbital workflow
- `/Users/aidanreddy/Desktop/VMC/notebooks/demo_gaussian.ipynb`: Gaussian-orbital workflow

## Installation and dependencies

See `/Users/aidanreddy/Desktop/VMC/INSTALL.md`.

## Citation

If you use this code, please cite the associated paper. See `/Users/aidanreddy/Desktop/VMC/CITATION.cff`.

```bibtex
@article{reddy2025quantum,
  title={Quantum melting a Wigner crystal into Hall liquids},
  author={Reddy, Aidan P and Fu, Liang},
  journal={arXiv preprint arXiv:2508.21000},
  year={2025}
}
```
