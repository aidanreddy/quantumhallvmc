#BUILD DETERMINANT

function calc_D_gaussian(R::Array{Float64},orbital_inputs)::Array{ComplexF64}
    (orbitalfunc, orbitalfunc_pi, orbitalfunc_pisqare, orbitalfunc_d_l0, gaussiansites) = orbital_inputs
    D=zeros(ComplexF64,(length(gaussiansites),size(R,1))) #allow to only compute for one particle at a time
    for i in eachindex(gaussiansites), j in axes(R,1)
        D[i,j] = orbitalfunc(gaussiansites[i],R[j,:])
    end
    return D
end

# DERIVATIVE WITH RESPECT TO VARIATIONAL PARAMETER: GAUSSIAN LENGTH SCALE L0

function calc_D_dl0_gaussian(R::Array{Float64},orbital_inputs)::Array{ComplexF64}
    (orbitalfunc, orbitalfunc_pi, orbitalfunc_pisqare, orbitalfunc_d_l0, gaussiansites) = orbital_inputs
    D_d_l0 = zeros(ComplexF64,(length(gaussiansites),size(R,1)))
    for i in eachindex(gaussiansites), j in axes(R,1)
        D_d_l0[i,j] = orbitalfunc_d_l0(gaussiansites[i],R[j,:])
    end
    return D_d_l0
end

# KINETIC ENERGY

function calc_D_pi_gaussian(R::Array{Float64},orbital_inputs)::Tuple{Array{ComplexF64},Array{ComplexF64}}
    (orbitalfunc, orbitalfunc_pi, orbitalfunc_pisqare, orbitalfunc_d_l0, gaussiansites) = orbital_inputs
    D_pix = zeros(ComplexF64,(length(gaussiansites),size(R,1))) #allow to only compute for one particle at a time
    d_piy = copy(D_pix)
    for i in eachindex(gaussiansites), j in axes(R,1)
        D_pix[i,j], d_piy[i,j] = orbitalfunc_pi(gaussiansites[i],R[j,:])
    end
    return D_pix, d_piy
end

function calc_D_pisquare_gaussian(R::Array{Float64},orbital_inputs)::Array{ComplexF64}
    (orbitalfunc, orbitalfunc_pi, orbitalfunc_pisqare, orbitalfunc_d_l0, gaussiansites) = orbital_inputs
    D_pisquare = zeros(ComplexF64,(length(gaussiansites),size(R,1)))
    for i in eachindex(gaussiansites), j in axes(R,1)
        D_pisquare[i,j] = orbitalfunc_pisqare(gaussiansites[i],R[j,:])
    end
    return D_pisquare
end

#HELPER FUNCTIONS FOR GAUSSIAN INITIALIZATION IN LL OR PLANEWAVE BASIS

function compute_overlap_k_b0_gaussian(mesh, RL, rs::Real, nu::Int64)
    C_Drummond = 0.15 * rs^(1/2) * nu / 2 # Supplementa Material of https://doi.org/10.1103/PhysRevLett.102.126402
    nk = length(mesh)
    overlap_k = zeros(ComplexF64, nk, length(RL))
    for i in eachindex(mesh), j in eachindex(RL)
        overlap_k[i, j] = exp(-(norm(mesh[i] .+ RL[j])^2 / (4 * C_Drummond)))
    end
    return overlap_k
end

function compute_overlap_k_b_gaussian(mesh, NLL::Int64,
                                      L1::AbstractVector{<:Real}, L2::AbstractVector{<:Real},
                                      rs::Real, nu::Int64, orbital_func_b::Function; numx::Int64=301)
    l_max = ceil(Int, max(20, max(norm(L1), norm(L2))))
    NL = 3 * l_max

    l_vals_cart = vec([i * L1 + j * L2 for i in -NL:NL, j in -NL:NL])
    l_vals = vec([[Int(i), Int(j)] for i in -NL:NL, j in -NL:NL])
    deleteat!(l_vals, findall(x -> (norm(x) > l_max), l_vals_cart))
    deleteat!(l_vals_cart, findall(x -> (norm(x) > l_max), l_vals_cart))

    # Drummond 2009 initialization. Supplemental Material of https://doi.org/10.1103/PhysRevLett.102.126402
    L0 = 0.5 / sqrt(0.15 * rs^(1/2) * nu / 2)

    nk = length(mesh)

    mag_gaussian = zeros(ComplexF64, (numx, numx))
    mag_bloch = zeros(ComplexF64, (nk, NLL, numx, numx))

    for i in 1:numx, j in 1:numx
        r = ((i - 1) * L1 + (j - 1) * L2) / numx - (L1 + L2) / 2
        mag_gaussian[i, j] = gaussian_mbc(r, L0, 1., l_vals, L1, L2)
        for k in 1:nk
            mag_bloch[k, :, i, j] .= orbital_func_b(k, r)[1:end-1]
        end
    end

    for k in 1:nk, n in 1:NLL
        mag_bloch[k, n, :, :] ./= sqrt(sum(abs.(mag_bloch[k, n, :, :]).^2))
    end

    overlap_k = zeros(ComplexF64, nk, NLL)
    for k in 1:nk
        for n in 1:NLL
            overlap_k[k, n] = sum(conj(mag_bloch[k, n, :, :]) .* mag_gaussian)
        end
        overlap_k[k, :] ./= sqrt(sum(abs.(overlap_k[k, :]).^2))
    end

    return overlap_k
end

function D_params_from_overlap_k_nu1(overlap_k::Array{ComplexF64,2}, n_band::Int64)
    nk, ncol = size(overlap_k)
    @assert ncol == n_band "second dimension of overlap_k must equal n_band"

    D_params = zeros(Float64, nk, 1, n_band-1, 2)

    for k in 1:nk
        ψ = copy(overlap_k[k, :])
        ψ ./= norm(ψ)

        θ = angle(ψ[1])
        ψ .*= exp(-1im * θ)
        a = real(ψ[1])
        @assert a ≥ 0
        b = ψ[2:end]
        b_norm = norm(b)

        if b_norm < 1e-12
            D_params[k, 1, :, 1] .= 0.0
            D_params[k, 1, :, 2] .= 0.0
            continue
        end

        t = atan(b_norm, a)
        v = (-1im * (t / b_norm)) .* b

        D_params[k, 1, :, 1] .= real.(v)
        D_params[k, 1, :, 2] .= imag.(v)
    end

    return D_params
end
