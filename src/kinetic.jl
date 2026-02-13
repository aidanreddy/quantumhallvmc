function calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)::ComplexF64
    #kinetic energy
    #determinant
    ne=size(D_inv)[1]
    pi_D_ratio=zeros(ComplexF64,(ne,2))
    pi_square_D_ratio=zeros(ComplexF64,ne)
    for i in 1:ne
        pi_D_ratio[i,1] = calc_D_ratio(pix_D[:,i],D_inv[i,:])
        pi_D_ratio[i,2] = calc_D_ratio(piy_D[:,i],D_inv[i,:])
        pi_square_D_ratio[i] = calc_D_ratio(pi_square_D[:,i],D_inv[i,:])
    end
    T_jastrow = (-1/2)*(sum(sum(gradx_u_pair_mat,dims=2).^2) + sum(sum(grady_u_pair_mat,dims=2).^2) + sum(laplacian_u_pair_mat))
    T_det = (1/2)*sum(pi_square_D_ratio)
    T_mix = (-1im)*(dot(vec(sum(gradx_u_pair_mat,dims=2)),pi_D_ratio[:,1]) + dot(vec(sum(grady_u_pair_mat,dims=2)),pi_D_ratio[:,2]))
    return  T_det + T_mix + T_jastrow
end
