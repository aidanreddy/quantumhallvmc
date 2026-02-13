using LinearAlgebra
using Interpolations

function gaussian(r::Vector{Float64}, L0::Float64)::ComplexF64
    return exp(-norm(r)^2 / L0^2 / 4) / sqrt(2π) / L0
end

# periodic boundary conditions

function gaussian_pbc(
    r::Vector{Float64},
    L0::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::ComplexF64
    result = 0.0 + 0im
    for L in lattice
        l_cart = L[1] * L1 + L[2] * L2
        result += gaussian(r + l_cart, L0)
    end
    return result
end

function calc_pi_gaussian_pbc(
    r::Vector{Float64},
    L0::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::Vector{ComplexF64}
    grad = zeros(ComplexF64, 2)
    for L in lattice
        l_cart = L[1] * L1 + L[2] * L2
        grad += (r + l_cart) * gaussian(r + l_cart, L0)
    end
    grad *= -1 / L0^2 / 2
    return - 1im * grad
end

function calc_pi_square_gaussian_pbc(
    r::Vector{Float64},
    L0::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::Float64
    laplacian = 0.0
    for L in lattice
        l_cart = L[1] * L1 + L[2] * L2
        laplacian += norm(r + l_cart)^2 * gaussian(r + l_cart, L0) / L0^4 / 4
        laplacian -= gaussian(r + l_cart, L0) / L0^2
    end
    return - laplacian
end

function calc_dl0_gaussian_pbc(
    r::Vector{Float64},
    L0::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::Float64
    dgd_l0 = 0.0
    for L in lattice
        l_cart = L[1] * L1 + L[2] * L2
        g = gaussian(r + l_cart, L0)
        dgd_l0 += (norm(r + l_cart)^2 / L0^3 / 2 - 1 / L0) * g
    end
    return dgd_l0
end

# magnetic boundary conditions (MBC)

function gaussian_mbc(
    r::Vector{Float64},
    L0::Float64,
    l::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::ComplexF64
    psi = 0.0 + 0im
    n_phi = (L1[1] * L2[2] - L1[2] * L2[1]) / (2 * π * l^2)
    for i in eachindex(lattice)
        L = lattice[i][1] * L1 + lattice[i][2] * L2
        theta = 
            (r[1] * L[2] - r[2] * L[1]) / l^2 / 2 +
            lattice[i][1] * lattice[i][2] * n_phi * π
        psi += gaussian(r + L, L0) * exp(1im * theta)
    end
    return psi
end

function calc_pi_dagger_pi_gaussian_m(
    r::Vector{Float64}, L0::Float64, l::Float64
)::ComplexF64
    z = r[1] + 1im * r[2]
    return gaussian(r, L0) * (z * conj(z) * l^(-2) * (1 + (l / L0)^2) - 4) * (1 - (l / L0)^2) /
           8 / l^2
end

function calc_pi_gaussian_m(r::Vector{Float64}, L0::Float64, l::Float64)::ComplexF64
    z = r[1] + 1im * r[2]
    return - 1im * z * gaussian(r, L0) * (1 - (l / L0)^2) / 2 / sqrt(2) / l^2
end

function calc_pi_dagger_gaussian_m(r::Vector{Float64}, L0::Float64, l::Float64)::ComplexF64
    zstar = r[1] - 1im * r[2]
    return 1im * zstar * gaussian(r, L0) * (1 + (l / L0)^2) / 2 / sqrt(2) / l^2
end

function calc_pi_gaussian_mbc(
    r::Vector{Float64},
    L0::Float64,
    l::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::Vector{ComplexF64}
    pi_psi = [0.0 + 0im, 0.0 + 0im]
    n_phi = (L1[1] * L2[2] - L1[2] * L2[1]) / (2 * π * l^2)
    for i in eachindex(lattice)
        L = lattice[i][1] * L1 + lattice[i][2] * L2
        theta = 
            (r[1] * L[2] - r[2] * L[1]) / l^2 / 2 + lattice[i][1] * lattice[i][2] * n_phi * π
        gaussian_pi = calc_pi_gaussian_m(r + L, L0, l)
        gaussian_pidagger = calc_pi_dagger_gaussian_m(r + L, L0, l)
        pi_psi .+=
            [gaussian_pi + gaussian_pidagger, 1im * (gaussian_pidagger - gaussian_pi)] .*
            exp(1im * theta) ./ sqrt(2)
    end
    return pi_psi
end

function calc_pi_square_gaussian_mbc(
    r::Vector{Float64},
    L0::Float64,
    l::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::ComplexF64
    #this is pi_x^2 + pi_y^2
    pi_squared_psi = 0.0 + 0im
    n_phi = (L1[1] * L2[2] - L1[2] * L2[1]) / (2 * π * l^2)
    for i in eachindex(lattice)
        L = lattice[i][1] * L1 + lattice[i][2] * L2
        theta = 
            (r[1] * L[2] - r[2] * L[1]) / l^2 / 2 + lattice[i][1] * lattice[i][2] * n_phi * π
        pi_squared_psi +=
            (2 * calc_pi_dagger_pi_gaussian_m(r + L, L0, l) + gaussian(r + L, L0) / l^2) *
            exp(1im * theta)
    end
    return pi_squared_psi
end

function calc_dl0_gaussian_mbc(
    r::Vector{Float64},
    L0::Float64,
    l::Float64,
    lattice::Vector{Vector{Int64}},
    L1::Vector{Float64},
    L2::Vector{Float64},
)::ComplexF64
    dpsid_l0 = 0.0 + 0im
    n_phi = (L1[1] * L2[2] - L1[2] * L2[1]) / (2 * π * l^2)
    for i in eachindex(lattice)
        L = lattice[i][1] * L1 + lattice[i][2] * L2
        theta = 
            (r[1] * L[2] - r[2] * L[1]) / l^2 / 2 + lattice[i][1] * lattice[i][2] * n_phi * π
        dpsid_l0 += (norm(r + L)^2 / L0^3 / 2 - 1 / L0) * gaussian(r + L, L0) * exp(1im * theta)
    end
    return dpsid_l0
end
