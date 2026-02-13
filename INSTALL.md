# Installation

## Requirements

- Julia 1.6+ (1.8+ recommended)

## Dependencies

These are defined in `Project.toml` and installed by `Pkg.instantiate()`. The list
includes the notebook packages so a single install covers scripts and notebooks.

- TensorOperations
- SpecialFunctions
- Interpolations
- Primes
- FFTW
- LinearAlgebra (stdlib)
- Statistics (stdlib)
- JLD2
- Plots
- ImageFiltering
- Random (stdlib)
- Printf (stdlib)

## Setup

From the repo root:

```bash
julia --project=.
```

In the Julia REPL:

```julia
import Pkg
Pkg.instantiate()
```
