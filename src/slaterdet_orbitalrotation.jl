function calc_C(
    n_band::Int64, nk::Int64, nu::Int64, d_params::Array{Float64}
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    @assert size(d_params) == (nk, nu, n_band - nu, 2)
    K = zeros(ComplexF64, (nk, n_band, n_band))
    K[:, 1:nu, (nu + 1):n_band] .= d_params[:, :, :, 1] .+ 1im * d_params[:, :, :, 2]
    K[:, (nu + 1):n_band, 1:nu] .= permutedims(
        d_params[:, :, :, 1] .- 1im * d_params[:, :, :, 2], [1, 3, 2]
    )
    C = zeros(ComplexF64, (nk, n_band, n_band))
    Λ = zeros(ComplexF64, (nk, n_band))
    M = copy(C)
    for k in 1:nk
        Ak = 1im * K[k, :, :]
        evalsk, Mk = eigen(Ak)
        # Gauge-fix eigenvectors so that the component with largest magnitude
        # in each eigenvector is real and non-negative. This removes arbitrary
        # complex phases and makes M reproducible.
        for j in axes(Mk, 2)
            col = Mk[:, j]
            idx = argmax(abs.(col))
            phase = angle(col[idx])
            Mk[:, j] .*= exp(-1im * phase)
        end
        Ck = exp(Ak)
        C[k, :, :] = Ck
        M[k, :, :] = Mk # M is unitary change of basis matrix such that Ak = M*Λ*M^†
        Λ[k, :] = evalsk # eigenvalues of Ak
    end
    return C, M, Λ
end

function calc_O(
    R::Array{Float64},
    orbital_inputs::Tuple{
        Int64, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Function, Bool, Bool
    },
)::Tuple{Array{ComplexF64}, Array{ComplexF64}}
    # Problem-specific basis-orbital evaluator (LL basis in this project).
    (n_band, mesh, RL, orbital_func, B_field) = orbital_inputs
    # O_padded includes one extra LL index used for kinetic energy.
    if B_field
        n_band_padded = n_band + 1
    else
        n_band_padded = n_band
    end
    O_padded = zeros(ComplexF64, (length(mesh), n_band_padded, size(R)[1]))
    for k_ind in eachindex(mesh), i in axes(R, 1)
        O_padded[k_ind, :, i] = orbital_func(k_ind, R[i, :])
    end
    O = O_padded[:, 1:n_band, :]
    return O, O_padded
end

function calc_D(C::Array{ComplexF64}, O::Array{ComplexF64}, nu::Int64)::Array{ComplexF64}
    D = zero.(O) # (k, state orbital, coordinate)
    for k in axes(D, 1), i in axes(D, 2), j in axes(D, 3), l in axes(C, 3)
        D[k, i, j] += C[k, i, l] * O[k, l, j]
    end
    return reshape(D[:, 1:nu, :], (size(D)[1] * nu, size(D)[3]))
end

#KINETIC ENERGY

function calc_O_kin_col(
    o_col::Array{ComplexF64},
    O_padded_col::Array{ComplexF64},
    orbital_inputs::Tuple{
        Int64, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Function, Bool, Bool
    },
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    (n_band, mesh, RL, orbital_func, B_field) = orbital_inputs
    if !B_field
        return calc_O_kin_col_b_zero(o_col, mesh, RL)
    else
        return calc_O_kin_col_b(o_col, O_padded_col, mesh, n_band)
    end
end

function calc_O_kin(
    O::Array{ComplexF64},
    O_padded::Array{ComplexF64},
    orbital_inputs::Tuple{
        Int64, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Function, Bool, Bool
    },
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    (n_band, mesh, RL, orbital_func, B_field) = orbital_inputs
    if !B_field
        return calc_O_kin_b_zero(O, mesh, RL)
    else
        return calc_O_kin_b(O, O_padded, mesh, n_band)
    end
end

function calc_O_kin_col_b(
    o_col::Array{ComplexF64},
    O_padded_col::Array{ComplexF64},
    mesh::Vector{Vector{Float64}},
    n_band::Int64,
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    a_o_col = zero.(o_col)
    adag_o_col = zero.(o_col)
    for k in eachindex(mesh)
        adag_o_col[k, 1] += O_padded_col[k, 2] #n=0 edge case
        for n in 1:(n_band - 1)
            i = n + 1
            adag_o_col[k, i] += sqrt(n + 1) * O_padded_col[k, i + 1]
            a_o_col[k, i] += sqrt(n) * O_padded_col[k, i - 1]
        end
    end
    pix_O_col = (a_o_col .+ adag_o_col) ./ √2
    piy_O_col = (a_o_col .- adag_o_col) ./ (√2 * 1im)
    #squared kinetic momentum
    pi_square_O_col = zero.(o_col)
    for n in 0:(n_band - 1)
        pi_square_O_col[:, n + 1] .+= 2 * n + 1
    end
    pi_square_O_col .*= o_col
    return pix_O_col, piy_O_col, pi_square_O_col
end

function calc_O_kin_b(
    O::Array{ComplexF64},
    O_padded::Array{ComplexF64},
    mesh::Vector{Vector{Float64}},
    n_band::Int64,
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    #gradient and pi^2 of orbital matrix O for kinetic energy
    #GRADIENT
    a_o = zero.(O)
    adag_o = zero.(O)
    ne = size(O)[end]
    for k in eachindex(mesh)
        for j in 1:ne
            adag_o[k, 1, j] += O_padded[k, 2, j] #n=0 edge case
            for n in 1:(n_band - 1)
                i = n + 1
                adag_o[k, i, j] += sqrt(n + 1) * O_padded[k, i + 1, j]
                a_o[k, i, j] += sqrt(n) * O_padded[k, i - 1, j]
            end
        end
    end
    pix_O = (a_o .+ adag_o) ./ √2
    piy_O = (a_o .- adag_o) ./ (√2 * 1im)
    #squared kinetic momentum
    pi_square_O = zero.(O)
    for n in 0:(n_band - 1)
        pi_square_O[:, n + 1, :] .+= 2 * n + 1
    end
    pi_square_O .*= O
    return pix_O, piy_O, pi_square_O
end

function calc_O_kin_col_b_zero(
    o_col::Array{ComplexF64}, mesh::Vector{Vector{Float64}}, RL::Vector{Vector{Float64}}
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    n_band = length(RL)
    pix_O_col = copy(o_col)
    piy_O_col = copy(o_col)
    pi_square_O_col = copy(o_col)
    for k in eachindex(mesh)
        for n in 1:n_band
            Q = mesh[k] + RL[n]
            pix_O_col[k, n] *= Q[1]
            piy_O_col[k, n] *= Q[2]
            pi_square_O_col[k, n] *= dot(Q, Q)
        end
    end
    return pix_O_col, piy_O_col, pi_square_O_col
end

function calc_O_kin_b_zero(
    O::Array{ComplexF64}, mesh::Vector{Vector{Float64}}, RL::Vector{Vector{Float64}}
)::Tuple{Array{ComplexF64}, Array{ComplexF64}, Array{ComplexF64}}
    n_band = length(RL)
    pix_O = copy(O)
    piy_O = copy(O)
    pi_square_O = copy(O)
    for k in eachindex(mesh)
        for n in 1:n_band
            Q = mesh[k] + RL[n]
            pix_O[k, n, :] .*= Q[1]
            piy_O[k, n, :] .*= Q[2]
            pi_square_O[k, n, :] .*= dot(Q, Q)
        end
    end
    return pix_O, piy_O, pi_square_O
end

#PARAMETER DERIVATIVES

function calc_dparams_C(
    n_band::Int64, nk::Int64, nu::Int64, U::Array{ComplexF64}, Λ::Array{ComplexF64}
)
    d_c_d_params = zeros(ComplexF64, (nk, nu, n_band, nu, n_band - nu, 2))
    G = zeros(ComplexF64, (nk, n_band, n_band))
    for k in 1:nk, i in 1:n_band, j in 1:n_band
        if Λ[k, i] == Λ[k, j]
            G[k, i, i] = exp(Λ[k, i])
        else
            G[k, i, j] = (exp(Λ[k, i]) - exp(Λ[k, j])) / (Λ[k, i] - Λ[k, j])
        end
    end
    VRe = zeros(ComplexF64, (n_band, n_band))
    VIm = copy(VRe)
    for k in 1:nk
        Uk = U[k, :, :]
        Gk = G[k, :, :]
        for I in 1:nu
            i = I
            for J in 1:(n_band - nu)
                j = J + nu
                VRe = 1im * (conj(Uk[i, :]) * transpose(Uk[j, :]) + conj(Uk[j, :]) * transpose(Uk[i, :])) #note extra factor of i because the matrix being exponentiated is not K, but 1im*K
                VIm = -1 * (conj(Uk[i, :]) * transpose(Uk[j, :]) - conj(Uk[j, :]) * transpose(Uk[i, :]))
                d_c_d_params[k, :, :, I, J, 1] = (Uk * (VRe .* Gk) * (Uk'))[1:nu, :]
                d_c_d_params[k, :, :, I, J, 2] = (Uk * (VIm .* Gk) * (Uk'))[1:nu, :]
            end
        end
    end
    return d_c_d_params
end

function calc_delta_orbital(
    D_inv::Matrix{ComplexF64},
    O::Array{ComplexF64},
    C::Array{ComplexF64},
    dDparams_C::Array{ComplexF64},
)::Array{ComplexF64}
    nk = size(C)[1]
    n_band = size(C)[3]
    nu = size(dDparams_C)[2]
    dC_log_detD = zeros(ComplexF64, (nk, nu, n_band))
    delta_orbital_raw = zeros(ComplexF64, (nk, nu, n_band - nu, 2))
    for k in axes(C, 1), l in axes(C, 3)
        for i in 1:nu
            dC_log_detD[k, i, l] = calc_D_ratio(O[k, l, :], D_inv[:, k + (i - 1) * nk])
        end
    end
    for k in axes(dDparams_C, 1), m in axes(dDparams_C, 2), n in axes(dDparams_C, 3)
        delta_orbital_raw[k, :, :, :] .+= dC_log_detD[k, m, n] .* dDparams_C[k, m, n, :, :, :]
    end
    return vec(delta_orbital_raw)
end
