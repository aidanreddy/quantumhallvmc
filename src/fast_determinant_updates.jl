#fast determinant updates based on Sherman-Morrison formula

function calc_D_ratio(
    D_new_coli::Vector{ComplexF64}, D_inv_rowi::Vector{ComplexF64}
)::ComplexF64
    #assumes D_new only differs from D in COLUMN i
    return dot(conj.(D_inv_rowi), D_new_coli) # dot conjugates its first argument
end

function update_D_inv(i, delta_D_new_coli, D_inv)::Matrix{ComplexF64}
    D_inv_new = 
        D_inv .-
        ((D_inv * delta_D_new_coli) * transpose(D_inv[i, :])) ./
        (1 + dot(conj.(D_inv[i, :]), delta_D_new_coli))
    return D_inv_new
end
