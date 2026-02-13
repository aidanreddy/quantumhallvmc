function calc_v_short(r::Array{Float64}, η::Float64)::Array{Float64}
    return erfc.(r ./ (2η)) ./ r
end

function calc_v_long_g(q::Array{Float64}, η::Float64)::Array{Float64}
    return 2π .* erfc.(η .* q) ./ q
end

function calc_Coulomb_energy_const(ne::Int64, η::Float64, area::Float64, l_vals, g_vals)
    #everything in units of e²/(ϵ*(length unit))
    #this is the constant part that doesnt depend on the configuration of particle coordinates
    #GVals excludes (0,0)
    npair = binomial(ne, 2)
    g_norms = norm.(g_vals)
    l_norms = norm.(l_vals)
    deleteat!(l_norms, findall(x -> x == 0.0, l_norms)) # remove self image singularity
    v_long_g = calc_v_long_g(g_norms, η)
    v_const_self = 
        (ne / 2) * (
            sum(calc_v_short(l_norms, η)) - 1 / (η * √π) + sum(v_long_g) / area -
            4 * √π * η / area
        )
    v_const_pair = -npair * (2π / area) * (2η / √π)
    return v_const_pair + v_const_self, v_long_g
end

function calc_Coulomb_energy(
    R::Array{Float64},
    η::Float64,
    area::Float64,
    v_long_g::Vector{Float64},
    v_const::Float64,
    l_vals,
    g_vals,
)
    #everything in units of e²/(ϵ*(length unit))
    #GVals excludes  (0,0)
    ne = size(R, 1)
    npair = binomial(ne, 2)
    rij_vals = [[0.0, 0.0] for _ in 1:npair]
    rij_minus_l_vals = [[0.0, 0.0] for _ in 1:(npair * length(l_vals))]
    cnt1 = 1
    cnt2 = 1
    for i in axes(R, 1), j in 1:(i - 1)
        rij = R[i, :] - R[j, :]
        rij_vals[cnt1] = rij
        cnt1 += 1
        for l in eachindex(l_vals)
            rij_minus_l_vals[cnt2] = rij - l_vals[l]
            cnt2 += 1
        end
    end
    v_short = sum(calc_v_short(norm.(rij_minus_l_vals), η))
    v_long = 0.0
    for i in eachindex(rij_vals), j in eachindex(g_vals)
        v_long += real(exp(1im * (dot(g_vals[j], rij_vals[i])))) * v_long_g[j]
    end
    v_long /= area
    return v_short + v_long + v_const
end

function calc_Coulomb_energy_change(
    change_ind::Int64,
    ri_old::Vector{Float64},
    R::Array{Float64},
    η::Float64,
    area::Float64,
    v_long_g::Vector{Float64},
    l_vals,
    g_vals,
)
    ne = size(R, 1)
    n_l = length(l_vals)
    # scratch storage for pairwise real-space and lattice-shifted separations
    rij_old_vals = [zeros(Float64, 2) for _ in 1:(ne - 1)]
    rij_new_vals = [zeros(Float64, 2) for _ in 1:(ne - 1)]
    rij_old_minus_l_vals = [zeros(Float64, 2) for _ in 1:((ne - 1) * n_l)]
    rij_new_minus_l_vals = [zeros(Float64, 2) for _ in 1:((ne - 1) * n_l)]
    return calc_Coulomb_energy_change!(
        change_ind,
        ri_old,
        R,
        η,
        area,
        v_long_g,
        l_vals,
        g_vals,
        rij_old_vals,
        rij_new_vals,
        rij_old_minus_l_vals,
        rij_new_minus_l_vals,
    )
end

function calc_Coulomb_energy_change!(
    change_ind::Int64,
    ri_old::Vector{Float64},
    R::Array{Float64},
    η::Float64,
    area::Float64,
    v_long_g::Vector{Float64},
    l_vals,
    g_vals,
    rij_old_vals::Vector{Vector{Float64}},
    rij_new_vals::Vector{Vector{Float64}},
    rij_old_minus_l_vals::Vector{Vector{Float64}},
    rij_new_minus_l_vals::Vector{Vector{Float64}},
)::Float64
    # change in coulomb energy upon moving only one particle
    # everything in units of e²/(ϵ*(length unit))
    # GVals excludes (0,0)
    ne = size(R, 1)
    @assert length(rij_old_vals) == ne - 1
    @assert length(rij_new_vals) == ne - 1
    @assert length(rij_old_minus_l_vals) == (ne - 1) * length(l_vals)
    @assert length(rij_new_minus_l_vals) == (ne - 1) * length(l_vals)

    # coordinates of old and new position
    ri_old1 = ri_old[1]
    ri_old2 = ri_old[2]
    ri_new1 = R[change_ind, 1]
    ri_new2 = R[change_ind, 2]

    cnt1 = 1
    cnt2 = 1
    @inbounds for j in 1:ne
        if j == change_ind
            continue
        end
        # separation vectors to particle j (old/new)
        rjx = R[j, 1]
        rjy = R[j, 2]

        rij_old = rij_old_vals[cnt1]
        rij_new = rij_new_vals[cnt1]

        dx_old = ri_old1 - rjx
        dy_old = ri_old2 - rjy
        dx_new = ri_new1 - rjx
        dy_new = ri_new2 - rjy

        rij_old[1] = dx_old
        rij_old[2] = dy_old
        rij_new[1] = dx_new
        rij_new[2] = dy_new

        # lattice-image shifted separations
        for l in eachindex(l_vals)
            L = l_vals[l]
            lx = L[1]
            ly = L[2]

            rij_old_l = rij_old_minus_l_vals[cnt2]
            rij_new_l = rij_new_minus_l_vals[cnt2]

            rij_old_l[1] = dx_old - lx
            rij_old_l[2] = dy_old - ly
            rij_new_l[1] = dx_new - lx
            rij_new_l[2] = dy_new - ly

            cnt2 += 1
        end

        cnt1 += 1
    end

    # short-range contribution (real space), inline version of calcvShort(norm(r), η)
    v_short_old = 0.0
    v_short_new = 0.0
    @inbounds for k in eachindex(rij_old_minus_l_vals)
        r_old = rij_old_minus_l_vals[k]
        R_new = rij_new_minus_l_vals[k]

        r_old_norm = sqrt(r_old[1] * r_old[1] + r_old[2] * r_old[2])
        R_new_norm = sqrt(R_new[1] * R_new[1] + R_new[2] * R_new[2])

        v_short_old += erfc(r_old_norm / (2η)) / r_old_norm
        v_short_new += erfc(R_new_norm / (2η)) / R_new_norm
    end

    # long-range contribution (reciprocal space), using cos(q·r) instead of exp(i q·r)
    v_long_old = 0.0
    v_long_new = 0.0
    @inbounds for j in eachindex(g_vals)
        G = g_vals[j]
        gx = G[1]
        gy = G[2]

        acc_old = 0.0
        acc_new = 0.0
        for i in eachindex(rij_old_vals)
            r_old = rij_old_vals[i]
            R_new = rij_new_vals[i]

            φOld = gx * r_old[1] + gy * r_old[2]
            φNew = gx * R_new[1] + gy * R_new[2]

            acc_old += cos(φOld)
            acc_new += cos(φNew)
        end

        v_long_old += acc_old * v_long_g[j]
        v_long_new += acc_new * v_long_g[j]
    end

    v_long_old /= area
    v_long_new /= area

    return (v_short_new + v_long_new) - (v_short_old + v_long_old)
end
