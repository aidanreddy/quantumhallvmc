# variational_energies.csv

Variational Monte Carlo (VMC) energies of optimized Slater-Jastrow wavefunctions for the fully spin-polarized two-dimensional electron gas (2DEG), computed across a range of interaction strengths, system sizes, magnetic field conditions, and wavefunction ansätze. These data underlie the phase diagrams presented in "Quantum melting a Wigner crystal into Hall liquids" (Reddy, 2025).

## Columns

| Column | Description |
|--------|-------------|
| `B_field` | Magnetic field condition: `nu=1` or `nu=2` (integer Landau level filling) or `B=0` (zero field) |
| `wavefunction_type` | Variational ansatz used (see below) |
| `N` | Number of electrons in the simulation cell |
| `r_s` | Dimensionless interaction strength: $r_s = 1/\sqrt{\pi a_B^2 n}$, where $a_B = \hbar^2/(e^2 m)$ is the Bohr radius and $n$ is the electron density |
| `(E-E_mad)*r_s^(3/2)_Ha` | Variational energy per particle, shifted by the Madelung energy and scaled by $r_s^{3/2}$, in Hartree. The actual energy per particle is $E = E_\mathrm{mad} + \mathrm{(this\ column)} / r_s^{3/2}$ |
| `std_err` | One standard error of the energy estimate, obtained by binning analysis of the local energy Monte Carlo time series |

The Madelung energy is $E_\mathrm{mad} = -1.106103\ \mathrm{Ha}/r_s$.

## Wavefunction types

- **Integer quantum Hall** — Slater-Jastrow with a Slater determinant built from the lowest Landau level single-particle orbitals; describes the quantum Hall liquid phase.
- **Landau level expansion** — Slater-Jastrow with orbitals expressed as a variational expansion over multiple Landau levels; describes the Wigner crystal phase.
- **Gaussian** — Slater-Jastrow with Gaussian orbitals centered on crystal lattice sites; an alternative Wigner crystal ansatz used for cross-checks.
- **Fermi liquid** — Slater-Jastrow with plane-wave orbitals filling the Fermi sea; zero-field liquid ansatz.
- **Planewave expansion** — Slater-Jastrow with a variational expansion of plane-wave orbitals; zero-field crystal ansatz.

## Notes

- All energies are per particle.
- The energy combination $(E - E_\mathrm{mad}) r_s^{3/2}$ is used because it is slowly varying in $r_s$, making trends easier to resolve numerically.
- Phase boundaries are estimated by linearly interpolating the liquid and crystal energies near the crossing point.
- At $\nu = 1$, the liquid-crystal transition is located near $r_s \approx 47$; at $\nu = 2$, near $r_s \approx 38$; at $B = 0$, near $r_s \approx 33$.
- Finite-size effects are assessed by comparing results at $N = 9, 36, 121, 144$ ($\nu=1$) and $N = 64$ ($\nu=2$).