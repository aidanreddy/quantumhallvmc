using LinearAlgebra

function cross_z_hat(V::Vector)::Vector{}
    return [V[2], -V[1]]
end

function geometry_setup(A1, A2, N1, N2, rl_cut, B_field, fermi_surface)
    L1 = N1 * A1
    L2 = N2 * A2
    area = abs(L1[1] * L2[2] - L1[2] * L2[1])
    G1 = 2pi * cross_z_hat(L2) / area
    G2 = -2pi * cross_z_hat(L1) / area
    g1 = 2pi * cross_z_hat(A2) / (area / (N1 * N2))
    g2 = -2pi * cross_z_hat(A1) / (area / (N1 * N2))
    mesh = reshape([i * G1 .+ j * G2 for i in 0:(N1 - 1), j in 0:(N2 - 1)], N1 * N2)
    if !B_field
        if fermi_surface
            mesh_raw = reshape(
                [i * G1 .+ j * G2 for i in (-N1):N1, j in (-N2):N2],
                (2 * N1 + 1) * (2 * N2 + 1),
            )
            mesh = mesh_raw[sortperm(norm.(mesh_raw))[1:(N1 * N2)]]
        else
            mesh_raw = copy(mesh)
            for k in eachindex(mesh), i in - 3:3, j in - 3:3
                if norm(mesh_raw[k] + i * g1 + j * g2) < norm(mesh[k])
                    mesh[k] = mesh_raw[k] + i * g1 + j * g2
                end
            end
        end
    end
    RL = circular_lattice(rl_cut, g1, g2)[2]
    RL = RL[sortperm(norm.(RL))]
    return area, L1, L2, G1, G2, mesh, g1, g2, RL
end

function send_to_first_supercell(R, supercell_to_cart, cart_to_supercell)::Vector{Float64}
    RSC = cart_to_supercell * R
    RSC .-= round.(RSC)
    return supercell_to_cart * RSC
end

function circular_lattice(max, a1, a2)
    n1 = Int64(2 * ceil(max / norm(a1)))
    n2 = Int64(2 * ceil(max / norm(a2)))
    avalscart = vec([i * a1 + j * a2 for i in (-n1):n1, j in (-n2):n2])
    avals = vec([[i, j] for i in (-n1):n1, j in (-n2):n2])
    deleteat!(avals, findall(x -> (norm(x) > max), avalscart))
    deleteat!(avalscart, findall(x -> (norm(x) > max), avalscart))
    return avals, avalscart
end
