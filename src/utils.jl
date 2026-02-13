function random_unitary_matrix(N::Int)
    x = (rand(N, N) + rand(N, N) * im) / sqrt(2)
    f = qr(x)
    diag_r = sign.(real(diag(f.R)))
    diag_r[diag_r .== 0] .= 1
    diag_rm = diagm(diag_r)
    u = f.Q * diag_rm
    return u
end

function calc_r_diffs(R::Array{Float64})::Matrix{Vector{Float64}}
    ne = size(R, 1)
    r_diffs = [[0.0, 0.0] for i in 1:ne, j in 1:ne]
    for i in 1:ne, j in 1:ne
        #RDiffs[i,j] .= R[i,:].-R[j,:]
        r_diffs[i, j] = R[i, :] - R[j, :]
    end
    return r_diffs
end

function update_r_diffs(i, ri_old, ri_new, r_diffs)::Array{Vector{Float64}}
    deltari = ri_new - ri_old
    r_diffs_updated = copy(r_diffs) #for some reason I cant just directly update RDiffs... must be some weird namespace stuff
    for j in axes(r_diffs, 1)
        if j == i
            continue
        end
        r_diffs_updated[i, j] += deltari
        r_diffs_updated[j, i] -= deltari
    end
    return r_diffs_updated
end
