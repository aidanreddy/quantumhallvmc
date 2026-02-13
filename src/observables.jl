function calc_structure_factor(
    R_walk,
    accept_walk::Vector{Int},
    particle_move_walk::Vector{Int},
    q_max::Float64,
    G1::Vector{Float64},
    G2::Vector{Float64},
)
    max = 2 * Int(floor(q_max / norm(G1)))
    q_vals_raw = [n * G1 + m * G2 for n in (-max):max, m in (-max):max]
    q_vals = [q for q in q_vals_raw if norm(q) < q_max]
    nq = length(q_vals)
    ne = size(R_walk[1], 1)
    rhoq = zeros(ComplexF64, nq)
    sq = zeros(Float64, nq)
    for r_ind in eachindex(R_walk)
        R = R_walk[r_ind]
        if r_ind == 1
            for i in 1:ne
                r_i = R[i, :]
                @inbounds for j in 1:nq
                    rhoq[j] += cis(dot(q_vals[j], r_i))
                end
            end
        else
            if accept_walk[r_ind] == 1
                i = particle_move_walk[r_ind]
                rprev = R_walk[r_ind - 1]
                R_new = R[i, :]
                r_old = rprev[i, :]
                @inbounds for j in 1:nq
                    q = q_vals[j]
                    rhoq[j] += cis(dot(q, R_new))
                    rhoq[j] -= cis(dot(q, r_old))
                end
            end
        end
        sq .+= abs2.(rhoq)
    end
    sq ./= (ne * length(R_walk))
    return sq, q_vals
end

function calc_pair_correlation(
    R_walk,
    particle_move_walk::Vector{Int64},
    ngrid::Int64,
    L1::Vector{Float64},
    L2::Vector{Float64},
)
    corners = [L1 L2 (L1 + L2) [0, 0]] .- (L1 + L2) / 2
    (x_min, x_max, y_min, y_max) = (
        minimum(corners[1, :]),
        maximum(corners[1, :]),
        minimum(corners[2, :]),
        maximum(corners[2, :]),
    )
    aspect = (y_max - y_min) / (x_max - x_min)
    ny = Int64(ceil(sqrt(ngrid * aspect)))
    nx = Int64(ceil(ngrid / ny))
    bins = zeros(Float64, (nx, ny))
    x_vals = LinRange(x_min, x_max, nx)
    y_vals = LinRange(y_min, y_max, ny)
    dx = step(x_vals)
    dy = step(y_vals)
    ne = size(R_walk[1], 1)
    supercell_to_cart = [L1 L2]
    cart_to_supercell = inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    rij_cart = [0.0, 0.0] #preallocate
    len_walk = length(R_walk)
    for r_ind in eachindex(R_walk)
        R = R_walk[r_ind]
        if r_ind == 1
            for i in 1:ne, j in 1:ne
                if i == j
                    continue
                end
                rij_cart = send_to_first_supercell(
                    R[i, :] .- R[j, :], supercell_to_cart, cart_to_supercell
                )
                bins[Int(cld(rij_cart[1] - x_min, dx)), Int(cld(rij_cart[2] - y_min, dy))] +=
                    min(
                        mod(j - particle_move_walk[1], ne), mod(i - particle_move_walk[1], ne)
                    ) + 1 #min(j,i)
            end
        else
            i = particle_move_walk[r_ind]
            for j in 1:ne
                if j != i
                    rij_cart = send_to_first_supercell(
                        R[i, :] .- R[j, :], supercell_to_cart, cart_to_supercell
                    )
                    #ij
                    bins[Int(cld(rij_cart[1] - x_min, dx)), Int(cld(rij_cart[2] - y_min, dy))] += min(
                        mod(j - i, ne), len_walk - r_ind + 1
                    )
                    #ji
                    bins[Int(cld(-rij_cart[1] - x_min, dx)), Int(cld(-rij_cart[2] - y_min, dy))] += min(
                        mod(j - i, ne), len_walk - r_ind + 1
                    )
                end
            end
        end
    end
    pair_corr = bins #./ (area/density^2)
    return pair_corr, x_vals, y_vals, aspect
end

function calc_density(
    R_walk,
    particle_move_walk::Vector{Int64},
    ngrid::Int64,
    L1::Vector{Float64},
    L2::Vector{Float64},
)
    corners = [L1 L2 (L1 + L2) [0, 0]] .- (L1 + L2) / 2
    (x_min, x_max, y_min, y_max) = (
        minimum(corners[1, :]),
        maximum(corners[1, :]),
        minimum(corners[2, :]),
        maximum(corners[2, :]),
    )
    aspect = (y_max - y_min) / (x_max - x_min)
    ny = Int64(ceil(sqrt(ngrid * aspect)))
    nx = Int64(ceil(ngrid / ny))
    bins = zeros(Float64, (nx, ny))
    x_vals = LinRange(x_min, x_max, nx)
    y_vals = LinRange(y_min, y_max, ny)
    dx = step(x_vals)
    dy = step(y_vals)
    ne = size(R_walk[1], 1)
    supercell_to_cart = [L1 L2]
    cart_to_supercell = inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    r_cart = [0.0, 0.0] #preallocate
    len_walk = length(R_walk)
    for r_ind in eachindex(R_walk)
        R = R_walk[r_ind]
        if r_ind == 1
            for i in 1:ne
                r_cart = send_to_first_supercell(
                    R[i, :], supercell_to_cart, cart_to_supercell
                )
                bins[Int(cld(r_cart[1] - x_min, dx)), Int(cld(r_cart[2] - y_min, dy))] +=
                    mod(i - particle_move_walk[1], ne) + 1
            end
        else
            i = particle_move_walk[r_ind]
            r_cart = send_to_first_supercell(R[i, :], supercell_to_cart, cart_to_supercell)
            bins[Int(cld(r_cart[1] - x_min, dx)), Int(cld(r_cart[2] - y_min, dy))] += min(
                ne, len_walk - r_ind + 1
            )
        end
    end
    density = bins #./ (area/density^2)
    return density, x_vals, y_vals, aspect
end
