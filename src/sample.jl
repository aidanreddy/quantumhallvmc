function sample(ne,R_start,walk_length,σ,C,orbital_inputs::Tuple{Int64,Vector{Vector{Float64}},Vector{Vector{Float64}},Function,Bool,Bool},nu,L1,L2,G1,G2,u_func,udx_func,udy_func,ulaplacian_func,comp_el)
    #### setup
    #for shifting corrdinates to first supercell
    supercell_to_cart=[L1 L2]
    cart_to_supercell=inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    R=copy(R_start) #Matrix(transpose((SCToCart*transpose(RStart))))
    r_diffs=calc_r_diffs(R)
    R_walk=[zeros(Float64,size(R)) for _ in 1:(walk_length+1)]
    R_walk[1]=R
    Tl_walk=[0. + 0im for _ in 1:(walk_length)]
    Vl_walk=copy(Tl_walk)
    accept_walk=[0 for _ in 1:(walk_length)]
    particle_move_walk=[0 for _ in 1:(walk_length)]
    #initialize wavefunction
    O,O_padded=calc_O(R,orbital_inputs) #OPadded includes one extra LL index necessary for computing kinetic energy
    @assert sum(isnan.(O_padded))==0     
    D=calc_D(C,O,nu)
    D_inv=inv(D)
    # initalize coulomb energy
    area=L1[1]L2[2]-L1[2]L2[1]
    η=norm(L1)/3.8 #3.8 is optimized to give same number of G's and L's with given erfcArgMax #I have played around with eta and NL, NG and convinced myself that current setup is a good solution for efficiency and accuracy (02/06/24)
    erfc_arg_max=6.0 #I verified that this seems sufficient for convergence to 1 part in 1e-8
    l_vals = circular_lattice((2η * erfc_arg_max),L1,L2)[2]
    g_vals = circular_lattice((erfc_arg_max / η),G1,G2)[2]
    deleteat!(g_vals, findall(x->x==[0.,0.],g_vals))
    u_pair_mat=zeros(Float64,(ne,ne))
    gradx_u_pair_mat=copy(u_pair_mat)
    grady_u_pair_mat=copy(u_pair_mat)
    laplacian_u_pair_mat=copy(u_pair_mat)
    for i in axes(r_diffs,1)
        u_pair_mat[i,:] = calc_U_row(i,r_diffs[i,:],u_func)
        gradx_u_pair_row, grady_u_pair_row = calc_dU_row(i,r_diffs[i,:],udx_func,udy_func)
        gradx_u_pair_mat[i,:] = gradx_u_pair_row
        grady_u_pair_mat[i,:] = grady_u_pair_row
        laplacian_u_pair_mat[i,:] = calc_laplacian_U_row(i,r_diffs[i,:],ulaplacian_func)
    end
    U_total=sum(u_pair_mat)/2
    pix_O,piy_O,pi_square_O=calc_O_kin(O,O_padded,orbital_inputs)
    pix_D=calc_D(C,pix_O,nu)
    piy_D=calc_D(C,piy_O,nu)
    pi_square_D=calc_D(C,pi_square_O,nu)
    if comp_el==true
        T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
        v_const,v_long_g=calc_coulomb_energy_const(ne,η,area,l_vals,g_vals)
        V=calc_coulomb_energy(R,η,area,v_long_g,v_const,l_vals,g_vals)
    end
    # markov chain
    reset_period=100*ne
    for I in 1:walk_length
        #dynamically adjust σ based on acceptance rate
        if mod(I-1,1000)==0 && I>1
            accept_rate= sum(accept_walk[I-999:I])/1000
            if accept_rate > .5
                σ *= 1.05
            elseif accept_rate < .5
                σ *= 0.95
            end
        end
        @assert sum(isnan.(O_padded))==0
        if mod(I,reset_period)==0
            O,O_padded=calc_O(R,orbital_inputs)
            D=calc_D(C,O,nu)
            D_inv=inv(D)
        end
        i=mod(I-1,ne)+1
        random_shift = σ*[randn(),randn()]
        shift = random_shift
        ri_new_raw = R[i,:] + shift
        ri_new=send_to_first_supercell(R[i,:] + shift,supercell_to_cart,cart_to_supercell) #shift
        o_new_coli, O_padded_new_coli=calc_O(reshape(ri_new,(1,2)),orbital_inputs)
        D_new_coli=calc_D(C,o_new_coli,nu)
        delta_D_new_coli=D_new_coli -D[:,i]
        slater_ratio=calc_D_ratio(vec(D_new_coli),D_inv[i,:])
        @assert isnan(slater_ratio)==false
        R_new=copy(R)            
        R_new[i,:]=ri_new
        r_diffs_new=update_r_diffs(i,R[i,:],ri_new,r_diffs)
        u_pair_mat_new=copy(u_pair_mat)
        u_pair_row_new=calc_U_row(i,r_diffs_new[i,:],u_func)
        u_pair_mat_new[i,:]=u_pair_row_new
        u_pair_mat_new[:,i]=u_pair_row_new
        U_total_new=sum(u_pair_mat_new)/2
        J_ratio=exp(U_total_new-U_total) #avoid blowing up issues working directly with J with U_total is large
        @assert isnan(J_ratio)==false
        acceptance_ratio=abs(slater_ratio*J_ratio)^2 
        pix_O_new_coli, piy_O_new_coli, pi_square_O_new_coli = calc_O_kin_col(o_new_coli,O_padded_new_coli,orbital_inputs)
        pix_D_new_coli=calc_D(C,pix_O_new_coli,nu)
        piy_D_new_coli=calc_D(C,piy_O_new_coli,nu)
        gradx_u_pair_row_new_i, grady_u_pair_row_new_i = calc_dU_row(i,r_diffs_new[i,:],udx_func,udy_func)
        D_inv_new = update_Dinv(i,delta_D_new_coli,D_inv) #Sherman-Morrison
        accept=(rand()<acceptance_ratio)
        if accept==true
            r_diffs=copy(r_diffs_new)
            #update slater stuff
            D_inv=update_Dinv(i,delta_D_new_coli,D_inv) #Sherman-Morrison
            D[:,i]=D_new_coli
            O[:,:,i]=o_new_coli
            O_padded[:,:,i]=O_padded_new_coli
            pix_O_new_coli, piy_O_new_coli, pi_square_O_new_coli = calc_O_kin_col(o_new_coli,O_padded_new_coli,orbital_inputs)
            pix_O[:,:,i] = pix_O_new_coli
            piy_O[:,:,i] = piy_O_new_coli
            pi_square_O[:,:,i] = pi_square_O_new_coli
            pix_D[:,i]=calc_D(C,pix_O_new_coli,nu)
            piy_D[:,i]=calc_D(C,piy_O_new_coli,nu)
            pi_square_D[:,i]=calc_D(C,pi_square_O_new_coli,nu)
            u_pair_mat=copy(u_pair_mat_new)
            gradx_u_pair_row, grady_u_pair_row = calc_dU_row(i,r_diffs_new[i,:],udx_func,udy_func)
            gradx_u_pair_mat[i,:] = gradx_u_pair_row
            grady_u_pair_mat[i,:] = grady_u_pair_row
            gradx_u_pair_mat[:,i] = -gradx_u_pair_row
            grady_u_pair_mat[:,i] = -grady_u_pair_row
            laplacian_u_pair_row=calc_laplacian_U_row(i,r_diffs_new[i,:],ulaplacian_func) #update both rows and columns of U_total matrices!!!!
            laplacian_u_pair_mat[i,:] = laplacian_u_pair_row
            laplacian_u_pair_mat[:,i] = laplacian_u_pair_row
            R=copy(R_new) #i dont know why I need to do this but if i just do R[i,:].=riNew i get bugs...
            if comp_el==true
                delta_v=calc_coulomb_energy_change(i,R_walk[I][i,:],R,η,area,v_long_g,l_vals,g_vals)
                T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
            end
            U_total=copy(U_total_new)
        else
            delta_v=0.
        end
        V+=delta_v
        accept_walk[I] = Int(accept)
        particle_move_walk[I] = i
        R_walk[I+1] = R
        if comp_el==true
            Vl_walk[I] = V
            Tl_walk[I] = T
        end
    end
    #remove initial state from RWalk
    return R_walk[2:end], Vl_walk, Tl_walk, accept_walk, particle_move_walk
end

function sample_opt(κ,ne,R_start,burn,walk_length,σ,C,orbital_inputs::Tuple{Int64,Vector{Vector{Float64}},Vector{Vector{Float64}},Function,Bool,Bool},nu,L1,L2,G1,G2,u_func,udx_func,udy_func,ulaplacian_func,ugradα_func,D_cd_D_params,NJ,ND)
    opt_J = (NJ != 0)
    opt_D = (ND != 0)
    NP=NJ+ND
    #### setup
    supercell_to_cart=[L1 L2]
    cart_to_supercell=inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    R=copy(R_start) #Matrix(transpose((SCToCart*transpose(RStart))))
    r_diffs=calc_r_diffs(R)
    r_prev = copy(R)
    delta=zeros(ComplexF64,NP) #updated at each step
    delta_prev=copy(delta)
    #quantities to be returned
    acceptance_rate = 0.
    Delta=zeros(ComplexF64,NP) #average to be returned at end
    delta_conj_delta=zeros(ComplexF64,(NP,NP))
    delta_conj_el=zeros(ComplexF64,NP)
    El_walk=zeros(ComplexF64,walk_length)
    v_avg = 0.
    t_avg = 0.
    #initialize wavefunction
    O,O_padded=calc_O(R,orbital_inputs) #OPadded includes one extra LL index necessary for computing kinetic energy
    @assert sum(isnan.(O_padded))==0     
    D=calc_D(C,O,nu)
    D_inv=inv(D)
    # initalize coulomb energy
    area=L1[1]L2[2]-L1[2]L2[1]
    erfc_arg_max=6.0 #I verified that this seems sufficient for convergence to 1 part in 1e-8
    η=norm(L1)/3.8 #3.8 is optimized to give same number of G's and L's with given erfcArgMax #I have played around with eta and NL, NG and convinced myself that current setup is a good solution for efficiency and accuracy (02/06/24)
    l_vals = circular_lattice((2η * erfc_arg_max),L1,L2)[2]
    g_vals = circular_lattice((erfc_arg_max / η),G1,G2)[2]
    deleteat!(g_vals, findall(x->x==[0.,0.],g_vals))
    v_const,v_long_g=calc_coulomb_energy_const(ne,η,area,l_vals,g_vals)
    V=calc_coulomb_energy(R,η,area,v_long_g,v_const,l_vals,g_vals)
    # preallocate Coulomb scratch buffers for single-particle moves
    n_l = length(l_vals)
    rij_old_vals        = [zeros(Float64,2) for _ in 1:(ne-1)]
    rij_new_vals        = [zeros(Float64,2) for _ in 1:(ne-1)]
    rij_old_minus_l_vals  = [zeros(Float64,2) for _ in 1:((ne-1)*n_l)]
    rij_new_minus_l_vals  = [zeros(Float64,2) for _ in 1:((ne-1)*n_l)]
    u_pair_mat=zeros(Float64,(ne,ne))
    gradx_u_pair_mat=copy(u_pair_mat)
    grady_u_pair_mat=copy(u_pair_mat)
    laplacian_u_pair_mat=copy(u_pair_mat)
    for i in axes(r_diffs,1)
        u_pair_mat[i,:] = calc_U_row(i,r_diffs[i,:],u_func)
        gradx_u_pair_row, grady_u_pair_row = calc_dU_row(i,r_diffs[i,:],udx_func,udy_func)
        gradx_u_pair_mat[i,:] = gradx_u_pair_row
        grady_u_pair_mat[i,:] = grady_u_pair_row
        laplacian_u_pair_mat[i,:] = calc_laplacian_U_row(i,r_diffs[i,:],ulaplacian_func)
    end
    U_total=sum(u_pair_mat)/2
    pix_O,piy_O,pi_square_O=calc_O_kin(O,O_padded,orbital_inputs)
    pix_D=calc_D(C,pix_O,nu)
    piy_D=calc_D(C,piy_O,nu)
    pi_square_D=calc_D(C,pi_square_O,nu)
    T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
    # markov chain
    reset_period=20*ne
    for I in 1:(walk_length+burn)
        @assert sum(isnan.(O_padded))==0
        @assert maximum(abs.(O_padded)) < 10
        if mod(I,reset_period)==0
            D=calc_D(C,O,nu)
            D_inv=inv(D)
        end
        i=mod(I-1,ne)+1
        random_shift = σ*[randn(),randn()]
        shift = random_shift
        ri_new_raw = R[i,:] + shift
        ri_new=send_to_first_supercell(ri_new_raw,supercell_to_cart,cart_to_supercell) #shift
        o_new_coli, O_padded_new_coli=calc_O(reshape(ri_new,(1,2)),orbital_inputs)
        D_new_coli=calc_D(C,o_new_coli,nu)
        delta_D_new_coli=D_new_coli -D[:,i]
        slater_ratio=calc_D_ratio(vec(D_new_coli),D_inv[i,:])
        @assert isnan(slater_ratio)==false
        R_new=copy(R)            
        R_new[i,:]=ri_new
        r_diffs_new=update_r_diffs(i,R[i,:],ri_new,r_diffs)
        u_pair_mat_new=copy(u_pair_mat)
        u_pair_row_new=calc_U_row(i,r_diffs_new[i,:],u_func)
        u_pair_mat_new[i,:]=u_pair_row_new
        u_pair_mat_new[:,i]=u_pair_row_new
        U_total_new=sum(u_pair_mat_new)/2 #factor of 1/2 to avoid double counting pairs
        J_ratio=exp(U_total_new-U_total) #avoid blowing up issues working directly with J with U_total is large
        acceptance_ratio = abs(slater_ratio*J_ratio)^2
        pix_O_new_coli, piy_O_new_coli, pi_square_O_new_coli = calc_O_kin_col(o_new_coli,O_padded_new_coli,orbital_inputs)
        pix_D_new_coli=calc_D(C,pix_O_new_coli,nu)
        piy_D_new_coli=calc_D(C,piy_O_new_coli,nu)
        gradx_u_pair_row_new_i, grady_u_pair_row_new_i = calc_dU_row(i,r_diffs_new[i,:],udx_func,udy_func)
        D_inv_new = update_Dinv(i,delta_D_new_coli,D_inv) #Sherman-Morrison
        accept=(rand()<acceptance_ratio)
        acceptance_rate += accept
        @assert isnan(J_ratio)==false
        if accept==true
            r_diffs=copy(r_diffs_new)
            #update slater stuff
            D_inv=copy(D_inv_new)
            D[:,i]=D_new_coli
            O[:,:,i]=o_new_coli
            O_padded[:,:,i]=O_padded_new_coli
            pix_O[:,:,i] = pix_O_new_coli
            piy_O[:,:,i] = piy_O_new_coli
            pi_square_O[:,:,i] = pi_square_O_new_coli
            pix_D[:,i]=pix_D_new_coli
            piy_D[:,i]=piy_D_new_coli
            pi_square_D[:,i]=calc_D(C,pi_square_O_new_coli,nu)
            u_pair_mat=copy(u_pair_mat_new)
            gradx_u_pair_mat[i,:] = gradx_u_pair_row_new_i
            grady_u_pair_mat[i,:] = grady_u_pair_row_new_i
            gradx_u_pair_mat[:,i] = -gradx_u_pair_row_new_i
            grady_u_pair_mat[:,i] = -grady_u_pair_row_new_i
            laplacian_u_pair_row=calc_laplacian_U_row(i,r_diffs_new[i,:],ulaplacian_func) #update both rows and columns of U_total matrices!!!!
            laplacian_u_pair_mat[i,:] = laplacian_u_pair_row
            laplacian_u_pair_mat[:,i] = laplacian_u_pair_row
            R=copy(R_new)
            U_total=copy(U_total_new)
            delta_v=calc_coulomb_energy_change!(i,r_prev[i,:],R,η,area,v_long_g,l_vals,g_vals,
                                        rij_old_vals,rij_new_vals,rij_old_minus_l_vals,rij_new_minus_l_vals)
            T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
        else
            delta_v=0.
        end
        V += delta_v
        if I>burn
            #UPDATE Δ
            if (I-burn)==1
                if opt_J==true
                    for l in 2:size(R)[1], j in 1:(l-1)
                        delta[1:NJ] += ugradα_func(R[l,:] - R[j,:])
                    end
                end
                if opt_D==true
                    delta[(NJ+1):end] = calc_delta_orbital(D_inv,O,C,D_cd_D_params)
                end
            else
                delta=copy(delta_prev) #copy(ΔWalk[:,I-1])
                if accept==true #proceed if particle was moved
                    if opt_J==true
                        for j in 1:ne
                            if j!=i
                                delta[1:NJ] += ugradα_func(R[i,:] - R[j,:]) #no factor of 1/2 is necessary here 
                                delta[1:NJ] -= ugradα_func(r_prev[i,:] - r_prev[j,:])
                            end
                        end
                    end
                    if opt_D==true
                        delta[(NJ+1):end] = calc_delta_orbital(D_inv,O,C,D_cd_D_params)
                    end
                end
            end
            delta_prev=copy(delta)
            el = T + V * κ
            v_avg += V * κ
            t_avg += T
            El_walk[I-burn] = el
            Delta .+= delta
            conjdelta=conj.(delta)
            delta_conj_el .+= conjdelta .* el
            @tensor delta_conj_delta[i,j] += conjdelta[i] * delta[j] #this is causing a serious performance penalty. 26 Jun 2025. Uses less RAM but is slower than saving full walk history and computing S at the end.
        end
        r_prev=copy(R)
    end
    el_std_dev = sqrt(sum(abs.(El_walk .- sum(El_walk)/walk_length ).^2)/(walk_length-1))
    return Delta/walk_length, delta_conj_delta/walk_length, delta_conj_el/walk_length, El_walk, sum(El_walk)/walk_length, v_avg/walk_length, t_avg/walk_length, el_std_dev, acceptance_rate/walk_length, R
end

function sample_gaussian(ne,R_start,walk_length,σ,orbital_inputs,L1,L2,G1,G2,u_func,udx_func,udy_func,ulaplacian_func,comp_el)
    #### setup
    #for shifting corrdinates to first supercell
    supercell_to_cart=[L1 L2]
    cart_to_supercell=inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    R=copy(R_start)
    r_diffs=calc_r_diffs(R)
    R_walk=[zeros(Float64,size(R)) for _ in 1:(walk_length+1)]
    R_walk[1]=R
    Tl_walk=[0. + 0im for _ in 1:(walk_length)]
    Vl_walk=copy(Tl_walk)
    accept_walk=[0 for _ in 1:(walk_length)]
    particle_move_walk=[0 for _ in 1:(walk_length)]
    #initialize wavefunction
    D=calc_D_gaussian(R,orbital_inputs)
    D_inv=inv(D)
    # initalize coulomb energy
    area=L1[1]L2[2]-L1[2]L2[1]
    η=norm(L1)/3.8 #3.8 is optimized to give same number of G's and L's with given erfcArgMax #I have played around with eta and NL, NG and convinced myself that current setup is a good solution for efficiency and accuracy (02/06/24)
    erfc_arg_max=6.0 #I verified that this seems sufficient for convergence to 1 part in 1e-8
    l_vals = circular_lattice((2η * erfc_arg_max),L1,L2)[2]
    g_vals = circular_lattice((erfc_arg_max / η),G1,G2)[2]
    deleteat!(g_vals, findall(x->x==[0.,0.],g_vals))
    u_pair_mat=zeros(Float64,(ne,ne))
    for i in axes(r_diffs,1)
        u_pair_mat[i,:] = calc_U_row(i,r_diffs[i,:],u_func)
    end
    U_total = sum(u_pair_mat)/2
    pix_D, piy_D = calc_D_pi_gaussian(R,orbital_inputs)
    pi_square_D = calc_D_pisquare_gaussian(R,orbital_inputs)
    gradx_u_pair_mat=copy(u_pair_mat)
    grady_u_pair_mat=copy(u_pair_mat)
    laplacian_u_pair_mat=copy(u_pair_mat)
    if comp_el==true
        for i in axes(r_diffs,1)
            gradx_u_pair_row, grady_u_pair_row = calc_dU_row(i,r_diffs[i,:],udx_func,udy_func)
            gradx_u_pair_mat[i,:] = gradx_u_pair_row
            grady_u_pair_mat[i,:] = grady_u_pair_row
            laplacian_u_pair_mat[i,:] = calc_laplacian_U_row(i,r_diffs[i,:],ulaplacian_func)
        end
        T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
        v_const,v_long_g=calc_coulomb_energy_const(ne,η,area,l_vals,g_vals)
        V=calc_coulomb_energy(R,η,area,v_long_g,v_const,l_vals,g_vals)
        # preallocate Coulomb scratch buffers for single-particle moves
        n_l = length(l_vals)
        rij_old_vals        = [zeros(Float64,2) for _ in 1:(ne-1)]
        rij_new_vals        = [zeros(Float64,2) for _ in 1:(ne-1)]
        rij_old_minus_l_vals  = [zeros(Float64,2) for _ in 1:((ne-1)*n_l)]
        rij_new_minus_l_vals  = [zeros(Float64,2) for _ in 1:((ne-1)*n_l)]
    end
    # markov chain
    reset_period=100*ne
    for I in 1:walk_length
        #dynamically adjust σ based on acceptance rate
        if mod(I-1,1000)==0 && I>1
            accept_rate= sum(accept_walk[I-999:I])/1000
            if accept_rate > .5
                σ *= 1.05
            elseif accept_rate < .5
                σ *= 0.95
            end
        end
        if mod(I,reset_period)==0
            D_inv=inv(D)
        end
        i=mod(I-1,ne)+1
        random_shift = σ*[randn(),randn()]
        shift = random_shift
        ri_new_raw = R[i,:] + shift
        ri_new=send_to_first_supercell(R[i,:] + shift,supercell_to_cart,cart_to_supercell) #shift
        D_new_coli=calc_D_gaussian(reshape(ri_new,(1,2)),orbital_inputs)
        delta_D_new_coli=D_new_coli -D[:,i]
        slater_ratio=calc_D_ratio(vec(D_new_coli),D_inv[i,:])
        @assert isnan(slater_ratio)==false
        R_new=copy(R)            
        R_new[i,:]=ri_new
        r_diffs_new=update_r_diffs(i,R[i,:],ri_new,r_diffs)
        u_pair_mat_new=copy(u_pair_mat)
        u_pair_row_new=calc_U_row(i,r_diffs_new[i,:],u_func)
        u_pair_mat_new[i,:]=u_pair_row_new
        u_pair_mat_new[:,i]=u_pair_row_new
        U_total_new=sum(u_pair_mat_new)/2
        J_ratio=exp(U_total_new-U_total) #avoid blowing up issues working directly with J with U_total is large
        @assert isnan(J_ratio)==false
        acceptance_ratio=abs(slater_ratio*J_ratio)^2 
        pix_D_new_coli, piy_D_new_coli = calc_D_pi_gaussian(reshape(ri_new,(1,2)),orbital_inputs)
        pi_square_D_new_coli = calc_D_pisquare_gaussian(reshape(ri_new,(1,2)),orbital_inputs)
        gradx_u_pair_row_new_i, grady_u_pair_row_new_i = calc_dU_row(i,r_diffs_new[i,:],udx_func,udy_func)
        D_inv_new = update_Dinv(i,delta_D_new_coli,D_inv) #Sherman-Morrison
        accept=(rand()<acceptance_ratio)
        if accept==true
            r_diffs=copy(r_diffs_new)
            #update slater stuff
            D_inv=update_Dinv(i,delta_D_new_coli,D_inv) #Sherman-Morrison
            D[:,i]=D_new_coli
            pix_D[:,i]=pix_D_new_coli
            piy_D[:,i]=piy_D_new_coli
            pi_square_D[:,i]=pi_square_D_new_coli
            u_pair_mat=copy(u_pair_mat_new)
            gradx_u_pair_row, grady_u_pair_row = calc_dU_row(i,r_diffs_new[i,:],udx_func,udy_func)
            gradx_u_pair_mat[i,:] = gradx_u_pair_row
            grady_u_pair_mat[i,:] = grady_u_pair_row
            gradx_u_pair_mat[:,i] = -gradx_u_pair_row
            grady_u_pair_mat[:,i] = -grady_u_pair_row
            laplacian_u_pair_row=calc_laplacian_U_row(i,r_diffs_new[i,:],ulaplacian_func) #update both rows and columns of U_total matrices!!!!
            laplacian_u_pair_mat[i,:] = laplacian_u_pair_row
            laplacian_u_pair_mat[:,i] = laplacian_u_pair_row
            R=copy(R_new) #i dont know why I need to do this but if i just do R[i,:].=riNew i get bugs...
            if comp_el==true
                delta_v=calc_coulomb_energy_change!(i,R_walk[I][i,:],R,η,area,v_long_g,l_vals,g_vals,
                                            rij_old_vals,rij_new_vals,rij_old_minus_l_vals,rij_new_minus_l_vals)
                T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
            end
            U_total=copy(U_total_new)
        else
            delta_v=0.
        end
        if comp_el==true V+=delta_v end
        accept_walk[I] = Int(accept)
        particle_move_walk[I] = i
        R_walk[I+1] = R
        if comp_el==true
            Vl_walk[I] = V
            Tl_walk[I] = T
        end
    end
    #remove initial state from RWalk
    return R_walk[2:end], Vl_walk, Tl_walk, accept_walk, particle_move_walk
end

function sample_opt_gaussian(κ,R_start,burn,walk_length,σ,orbital_inputs,ne,L1,L2,G1,G2,u_func,udx_func,udy_func,ulaplacian_func,ugradα_func,NJ,ND)
    opt_J = (NJ != 0)
    opt_D = (ND != 0)
    NP=NJ+ND
    #### setup
    supercell_to_cart=[L1 L2]
    cart_to_supercell=inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    R=copy(R_start)
    r_diffs=calc_r_diffs(R)
    r_prev = copy(R)
    delta=zeros(ComplexF64,NP) #updated at each step
    delta_prev=copy(delta)
    #quantities to be returned
    acceptance_rate = 0.
    Delta=zeros(ComplexF64,NP) #average to be returned at end
    delta_conj_delta=zeros(ComplexF64,(NP,NP))
    delta_conj_el=zeros(ComplexF64,NP)
    El_walk=zeros(ComplexF64,walk_length)
    v_avg = 0.
    t_avg = 0.
    #initialize wavefunction
    D=calc_D_gaussian(R,orbital_inputs)
    D_d_l0 = zero.(D) #initialize dDdL0 and compute after burn
    D_inv=inv(D)
    # initalize coulomb energy
    area=L1[1]L2[2]-L1[2]L2[1]
    erfc_arg_max=6.0 #I verified that this seems sufficient for convergence to 1 part in 1e-8
    η=norm(L1)/3.8 #3.8 is optimized to give same number of G's and L's with given erfcArgMax #I have played around with eta and NL, NG and convinced myself that current setup is a good solution for efficiency and accuracy (02/06/24)
    l_vals = circular_lattice((2η * erfc_arg_max),L1,L2)[2]
    g_vals = circular_lattice((erfc_arg_max / η),G1,G2)[2]
    deleteat!(g_vals, findall(x->x==[0.,0.],g_vals))
    v_const,v_long_g=calc_coulomb_energy_const(ne,η,area,l_vals,g_vals)
    V=calc_coulomb_energy(R,η,area,v_long_g,v_const,l_vals,g_vals)
    u_pair_mat=zeros(Float64,(ne,ne))
    gradx_u_pair_mat=copy(u_pair_mat)
    grady_u_pair_mat=copy(u_pair_mat)
    laplacian_u_pair_mat=copy(u_pair_mat)
    for i in axes(r_diffs,1)
        u_pair_mat[i,:] = calc_U_row(i,r_diffs[i,:],u_func)
        gradx_u_pair_row, grady_u_pair_row = calc_dU_row(i,r_diffs[i,:],udx_func,udy_func)
        gradx_u_pair_mat[i,:] = gradx_u_pair_row
        grady_u_pair_mat[i,:] = grady_u_pair_row
        laplacian_u_pair_mat[i,:] = calc_laplacian_U_row(i,r_diffs[i,:],ulaplacian_func)
    end
    U_total=sum(u_pair_mat)/2
    pix_D, piy_D = calc_D_pi_gaussian(R,orbital_inputs)
    pi_square_D=calc_D_pisquare_gaussian(R,orbital_inputs)
    T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
    # markov chain
    reset_period=20*ne
    for I in 1:(walk_length+burn)
        if mod(I,reset_period)==0
            D_inv=inv(D)
        end
        i=mod(I-1,ne)+1
        random_shift = σ*[randn(),randn()]
        shift = random_shift
        ri_new_raw = R[i,:] + shift
        ri_new=send_to_first_supercell(ri_new_raw,supercell_to_cart,cart_to_supercell) #shift
        D_new_coli=calc_D_gaussian(reshape(ri_new,(1,2)),orbital_inputs)
        delta_D_new_coli=D_new_coli -D[:,i]
        slater_ratio=calc_D_ratio(vec(D_new_coli),D_inv[i,:])
        @assert isnan(slater_ratio)==false
        R_new=copy(R)            
        R_new[i,:]=ri_new
        r_diffs_new=update_r_diffs(i,R[i,:],ri_new,r_diffs)
        u_pair_mat_new=copy(u_pair_mat)
        u_pair_row_new=calc_U_row(i,r_diffs_new[i,:],u_func)
        u_pair_mat_new[i,:]=u_pair_row_new
        u_pair_mat_new[:,i]=u_pair_row_new
        U_total_new=sum(u_pair_mat_new)/2 #factor of 1/2 to avoid double counting pairs
        J_ratio=exp(U_total_new-U_total) #avoid blowing up issues working directly with J with U_total is large
        acceptance_ratio = abs(slater_ratio*J_ratio)^2
        pix_D_new_coli, piy_D_new_coli = calc_D_pi_gaussian(reshape(ri_new,(1,2)),orbital_inputs)
        gradx_u_pair_row_new_i, grady_u_pair_row_new_i = calc_dU_row(i,r_diffs_new[i,:],udx_func,udy_func)
        D_inv_new = update_Dinv(i,delta_D_new_coli,D_inv) #Sherman-Morrison
        accept=(rand()<acceptance_ratio)
        acceptance_rate += accept
        @assert isnan(J_ratio)==false
        if accept==true
            r_diffs=copy(r_diffs_new)
            #update slater stuff
            D_inv=copy(D_inv_new)
            D[:,i]=D_new_coli
            pix_D[:,i]=pix_D_new_coli
            piy_D[:,i]=piy_D_new_coli
            pi_square_D[:,i]=calc_D_pisquare_gaussian(reshape(ri_new,(1,2)),orbital_inputs)
            u_pair_mat=copy(u_pair_mat_new)
            gradx_u_pair_mat[i,:] = gradx_u_pair_row_new_i
            grady_u_pair_mat[i,:] = grady_u_pair_row_new_i
            gradx_u_pair_mat[:,i] = -gradx_u_pair_row_new_i
            grady_u_pair_mat[:,i] = -grady_u_pair_row_new_i
            laplacian_u_pair_row=calc_laplacian_U_row(i,r_diffs_new[i,:],ulaplacian_func) #update both rows and columns of U_total matrices!!!!
            laplacian_u_pair_mat[i,:] = laplacian_u_pair_row
            laplacian_u_pair_mat[:,i] = laplacian_u_pair_row
            R=copy(R_new) #i dont know why I need to do this but if i just do R[i,:].=riNew i get bugs...
            U_total=copy(U_total_new)
            delta_v=calc_coulomb_energy_change(i,r_prev[i,:],R,η,area,v_long_g,l_vals,g_vals)
            T=calc_kinetic_energy(pix_D,piy_D,pi_square_D,D_inv,gradx_u_pair_mat,grady_u_pair_mat,laplacian_u_pair_mat)
        else
            delta_v=0.
        end
        V += delta_v
        if I>burn
            #UPDATE Δ
            if (I-burn)==1
                if opt_J==true
                    for l in 2:size(R)[1], j in 1:(l-1)
                        delta[1:NJ] += ugradα_func(R[l,:] - R[j,:])
                    end
                end
                if opt_D==true
                    D_d_l0 = calc_D_dl0_gaussian(R,orbital_inputs)
                    delta[end] = tr(D_inv * D_d_l0) #using Jacobi formula
                end
            else
                delta=copy(delta_prev) #copy(ΔWalk[:,I-1])
                if accept==true #proceed if particle was moved
                    if opt_J==true
                        for j in 1:ne
                            if j!=i
                                delta[1:NJ] += ugradα_func(R[i,:] - R[j,:]) #no factor of 1/2 is necessary here 
                                delta[1:NJ] -= ugradα_func(r_prev[i,:] - r_prev[j,:])
                            end
                        end
                    end
                    if opt_D==true
                        D_d_l0[:,i] = calc_D_dl0_gaussian(reshape(ri_new,(1,2)), orbital_inputs)
                        delta[end] = tr(D_inv * D_d_l0) #using Jacobi formula
                    end
                end
            end
            delta_prev=copy(delta)
            el = T + V * κ
            v_avg += V * κ
            t_avg += T
            El_walk[I-burn] = el
            Delta .+= delta
            conjdelta=conj.(delta)
            delta_conj_el .+= conjdelta .* el
            @tensor delta_conj_delta[i,j] += conjdelta[i] * delta[j] #this is causing a serious performance penalty. 26 Jun 2025. Uses less RAM but is slower than saving full walk history and computing S at the end.
        end
        r_prev=copy(R)
    end
    el_std_dev = sqrt(sum(abs.(El_walk .- sum(El_walk)/walk_length ).^2)/(walk_length-1))
    return Delta/walk_length, delta_conj_delta/walk_length, delta_conj_el/walk_length, El_walk, sum(El_walk)/walk_length, v_avg/walk_length, t_avg/walk_length, el_std_dev, acceptance_rate/walk_length, R
end
