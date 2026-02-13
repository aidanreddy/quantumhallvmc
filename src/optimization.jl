#ORBITAL ROTATION

function opt_wf(opt_method::String,orbital_inputs::Tuple{Int64,Vector{Vector{Float64}},Vector{Vector{Float64}},Function,Bool,Bool},burn::Int64,μ::Float64,τ::Float64,ξ::Float64,ϵ::Float64,n_walkers::Int64,walk_length::Int64,n_opt::Int64,κ::Float64,σ::Float64,nu::Int64,N1::Int64,N2::Int64,A1::Vector{Float64},A2::Vector{Float64},rl_cut::Float64,J_params::Vector{Float64},D_params::Array{Float64},opt_J::Bool,opt_D::Bool)
    @assert opt_J==true || opt_D==true
    #setup
    nk=N1*N2
    ne=nu*nk
    (n_band,mesh,RL,orbital_func,B_field,fermi_surface) = orbital_inputs
    area, L1, L2, G1, G2, mesh, g1, g2, RL = geometry_setup(A1,A2,N1,N2,rl_cut,B_field,fermi_surface)
    γ=norm(L1)/sqrt(2)/π  * κ/3
    supercell_to_cart=[L1 L2]
    cart_to_supercell=inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    #initialize miscellaneous
    D_params_size=size(D_params)
    params=Vector{Float64}()
    if opt_J==true params=vcat(params,J_params) end
    if opt_D==true params=vcat(params,vec(D_params)) end
    e_opt_hist=zeros(Float64,n_opt)
    e_err_opt_hist=zeros(Float64,n_opt)
    t_opt_hist=zeros(Float64,n_opt)
    v_opt_hist=zeros(Float64,n_opt)
    params_opt_hist=zeros(Float64,(n_opt,length(params)))
    residual_opt_hist=zeros(Float64,(n_opt,length(params)))
    el_std_dev_opt_hist=zeros(Float64,n_opt)
    force_norm_opt_hist=zeros(Float64,n_opt)
    mc_time_avg=0.0
    sr_time_avg=0.0
    NJ=length(J_params)*opt_J
    ND=length(D_params)*opt_D
    n_params=NJ+ND
    #optimize
    R_start_list=[Matrix(transpose((supercell_to_cart*transpose(rand(ne,2) .- 0.5)))) for _ in 1:n_walkers]
    delta_list=[zeros(ComplexF64,n_params) for _ in 1:n_walkers]
    delta_conj_delta_list=[zeros(ComplexF64,(n_params,n_params)) for _ in 1:n_walkers]
    delta_conj_el_list=[zeros(ComplexF64,n_params) for _ in 1:n_walkers]
    el_list=[0. + 0im for _ in 1:n_walkers]
    El_walk_list=[zeros(ComplexF64,walk_length) for _ in 1:n_walkers]
    tl_list=[0. + 0im for _ in 1:n_walkers]
    vl_list=[0. for _ in 1:n_walkers]
    el_std_dev_list=[0. for _ in 1:n_walkers]
    acceptance_rate_list=[0. for _ in 1:n_walkers]
    r_stop_list=[zeros(Float64,(ne,2)) for _ in 1:n_walkers]
    f=zeros(ComplexF64,n_params) #force
    S=zeros(ComplexF64,(n_params,n_params)) #quantum metric
    C, M, Λ = calcC(n_band,nk,nu,D_params)
    D_cd_D_params = [exp(1im)] #dummy value for first optimization step
    δ = zeros(Float64,length(params)) #initialize parameter shift vector. 12 Jul 2025
    for i in 1:n_opt #optimization steps
        #build callable functions for a given parameter set α
        # Always use sinSpl Jastrow function
        c=vcat(J_params)
        #build Jastrow functions
        u_func(r) = calc_u_sin_spl(r,G1,G2,c,γ)
        udx_func(r) = calc_grad_u_sin_spl(r,G1,G2,c,γ)[1]
        udy_func(r) = calc_grad_u_sin_spl(r,G1,G2,c,γ)[2]
        ulaplacian_func(r) = calc_laplacian_u_sin_spl(r,G1,G2,c,γ)
        ugradα_func(r) = calc_grad_alpha_u_sin_spl(r,G1,G2,c)
        if opt_D==true
            C, M, Λ = calcC(n_band,nk,nu,D_params)
            D_cd_D_params=calcdCdDParams(n_band,nk,nu,M,Λ) #only consider part of C that matters for D
        end
        #run random walks
        tic=time()
        Threads.@threads for j in 1:n_walkers
            delta_list[j], delta_conj_delta_list[j], delta_conj_el_list[j], El_walk_list[j], el_list[j], vl_list[j], tl_list[j], el_std_dev_list[j], acceptance_rate_list[j], r_stop_list[j] = sample_opt(κ,ne,R_start_list[j],burn, walk_length ,σ,C,orbital_inputs,nu,L1,L2,G1,G2,u_func,udx_func,udy_func,ulaplacian_func,ugradα_func,D_cd_D_params,NJ,ND)
        end
        toc=time()
        mc_time = (toc-tic) * walk_length / (walk_length + burn)
        for i in 1:n_walkers R_start_list[i] = r_stop_list[i] end #each walker at each optimization step starts at last position of its previous optimization step walk
        accept_rate = sum(acceptance_rate_list)/n_walkers
        if accept_rate > .5 #.574 #settles on .4 after a bit of playing around
            σ *= 1.05
        elseif accept_rate < .5 #.574
            σ *= 0.95
        end
        #average over walkers
        delta_conj_delta=sum(delta_conj_delta_list)/n_walkers
        delta=sum(delta_list)/n_walkers
        delta_conj_el=sum(delta_conj_el_list)/n_walkers
        el=sum(el_list)/n_walkers
        El_walk=vcat(El_walk_list...)
        tl=sum(tl_list)/n_walkers
        vl=sum(vl_list)/n_walkers
        el_std_dev, error_bars = mc_stats(El_walk/ne)[[2,4]]
        ###
        f .= delta_conj_el - conj.(delta) * el #force
        S .= delta_conj_delta - delta' .* delta #quantum metric
        t_opt_hist[i]=real.(tl/ne)
        v_opt_hist[i]=real.(vl/ne)
        e_opt_hist[i]=real.(el/ne)
        e_err_opt_hist[i]=error_bars
        el_std_dev_opt_hist[i]=el_std_dev
        force_norm_opt_hist[i]=norm(f)
        params_opt_hist[i,:] .= params
        #update parameters though stochastic reconfiguration method
        tic=time()
        if i < n_opt #don't update params on last cycle so that the last set of params are sampled. 23 Jun 2025t
            if opt_method == "SR"
                stochasticreconfig!(δ,f,S,τ /(1+(ξ * i/n_opt)),ϵ,μ) #changed to modifying function. 12 Jul 2025.
                params .+= δ
            end
            if opt_J==true 
                J_params=params[1:NJ]
            end
            if opt_D==true D_params=reshape(params[(NJ+1):end],D_params_size) end
        end
        residual_opt_hist[i,:] .= δ
        toc=time()
        sr_time=toc-tic
        mc_time_avg += mc_time/n_opt
        sr_time_avg += sr_time/n_opt
        burn *= 0 #kill burn after first optimization cycle. 21 Aug 2025.
    end
    return C, J_params, D_params, e_opt_hist, t_opt_hist, v_opt_hist, e_err_opt_hist, el_std_dev_opt_hist, force_norm_opt_hist, params_opt_hist, mc_time_avg, sr_time_avg
end

#GAUSSIAN ORBITALS

function opt_wf_gaussian(opt_method::String,burn::Int64,μ::Float64,τ::Float64,ξ::Float64,ϵ::Float64,n_walkers::Int64,walk_length::Int64,n_opt::Int64,κ::Float64,σ::Float64,ne::Int64,B_field::Bool,N1::Int64,N2::Int64,A1::Vector{Float64},A2::Vector{Float64},L1::Vector{Float64}, L2::Vector{Float64}, G1::Vector{Float64}, G2::Vector{Float64}, l_vals::Vector{Vector{Int64}}, gaussiansites, J_params::Vector{Float64},D_params::Array{Float64},opt_J::Bool,opt_D::Bool)
    @assert opt_J==true || opt_D==true
    #setup
    supercell_to_cart=[L1 L2]
    cart_to_supercell=inv(supercell_to_cart) #cartesian to supercell coordinate basis change
    γ=norm(L1)/sqrt(2)/π  * κ/3
    #initialize miscellaneous
    D_params_size=size(D_params)
    params=Vector{Float64}()
    if opt_J==true params=vcat(params,J_params) end
    if opt_D==true params=vcat(params,vec(D_params)) end
    e_opt_hist=zeros(Float64,n_opt)
    e_err_opt_hist=zeros(Float64,n_opt)
    t_opt_hist=zeros(Float64,n_opt)
    v_opt_hist=zeros(Float64,n_opt)
    params_opt_hist=zeros(Float64,(n_opt,length(params)))
    residual_opt_hist=zeros(Float64,(n_opt,length(params)))
    el_std_dev_opt_hist=zeros(Float64,n_opt)
    force_norm_opt_hist=zeros(Float64,n_opt)
    mc_time_avg=0.0
    sr_time_avg=0.0
    #togglable opt params
    NJ=length(J_params)*opt_J
    ND=length(D_params)*opt_D
    n_params=NJ+ND
    #optimize
    R_start_list=[Matrix(transpose((supercell_to_cart*transpose(rand(ne,2) .- 0.5)))) for _ in 1:n_walkers]
    delta_list=[zeros(ComplexF64,n_params) for _ in 1:n_walkers]
    delta_conj_delta_list=[zeros(ComplexF64,(n_params,n_params)) for _ in 1:n_walkers]
    delta_conj_el_list=[zeros(ComplexF64,n_params) for _ in 1:n_walkers]
    el_list=[0. + 0im for _ in 1:n_walkers]
    El_walk_list=[zeros(ComplexF64,walk_length) for _ in 1:n_walkers]
    tl_list=[0. + 0im for _ in 1:n_walkers]
    vl_list=[0. for _ in 1:n_walkers]
    el_std_dev_list=[0. for _ in 1:n_walkers]
    acceptance_rate_list=[0. for _ in 1:n_walkers]
    r_stop_list=[zeros(Float64,(ne,2)) for _ in 1:n_walkers]
    f=zeros(ComplexF64,n_params) #force
    S=zeros(ComplexF64,(n_params,n_params)) #quantum metric
    δ = zeros(Float64,length(params)) #initialize parameter shift vector. 12 Jul 2025
    for i in 1:n_opt #optimization steps
        #build callable functions for a given parameter set α
        # Always use sinSpl Jastrow function
        c=vcat(J_params)
        L0 = D_params[1]
        #build Jastrow functions
        if B_field==false #periodic boundary conditions (B=0)
            orbital_func_pbc(R0,r) = gaussian_pbc(r + R0,L0,l_vals,L1,L2)
            orbital_func_pbc_pi(R0,r) = gaussian_pbc_pi(r + R0,L0,l_vals,L1,L2)
            orbital_func_pbc_pisqare(R0,r) = gaussian_pbc_pisquare(r + R0,L0,l_vals,L1,L2)
            orbital_func_pbc_d_l0(R0,r) = gaussian_pbc_dl0(r + R0,L0,l_vals,L1,L2)
            orbital_inputs = (orbital_func_pbc, orbital_func_pbc_pi, orbital_func_pbc_pisqare, orbital_func_pbc_d_l0, gaussiansites)
        elseif B_field==true #magnetic boundary conditions
            orbital_func_mbc(R0,r) = gaussian_mbc(r + R0,L0,1.,l_vals,L1,L2) * exp(1im * (r[1] * R0[2] - r[2] * R0[1]) / 2)
            orbital_func_mbc_pi(R0,r) = gaussian_mbc_pi(r + R0,L0,1.,l_vals,L1,L2) * exp(1im * (r[1] * R0[2] - r[2] * R0[1]) / 2)
            orbital_func_mbc_pisqare(R0,r) = gaussian_mbc_pisquared(r + R0,L0,1.,l_vals,L1,L2) * exp(1im * (r[1] * R0[2] - r[2] * R0[1]) / 2)
            orbital_func_mbc_d_l0(R0,r) = gaussian_mbc_dl0(r + R0,L0,1.,l_vals,L1,L2) * exp(1im * (r[1] * R0[2] - r[2] * R0[1]) / 2)
            orbital_inputs = (orbital_func_mbc, orbital_func_mbc_pi, orbital_func_mbc_pisqare, orbital_func_mbc_d_l0, gaussiansites)
        end
        u_func(r) = calc_u_sin_spl(r,G1,G2,c,γ)
        udx_func(r) = calc_grad_u_sin_spl(r,G1,G2,c,γ)[1]
        udy_func(r) = calc_grad_u_sin_spl(r,G1,G2,c,γ)[2]
        ulaplacian_func(r) = calc_laplacian_u_sin_spl(r,G1,G2,c,γ)
        ugradα_func(r) = calc_grad_alpha_u_sin_spl(r,G1,G2,c)
        #run random walks
        tic=time()
        Threads.@threads for j in 1:n_walkers
            delta_list[j], delta_conj_delta_list[j], delta_conj_el_list[j], El_walk_list[j], el_list[j], vl_list[j], tl_list[j], el_std_dev_list[j], acceptance_rate_list[j], r_stop_list[j] = sample_opt_gaussian(κ,R_start_list[j],burn,walk_length,σ,orbital_inputs,ne,L1,L2,G1,G2, u_func,udx_func,udy_func,ulaplacian_func,ugradα_func,NJ,ND)
        end
        toc=time()
        mc_time = (toc-tic) * walk_length / (walk_length + burn)
        for i in 1:n_walkers R_start_list[i] = r_stop_list[i] end #each walker at each optimization step starts at last position of its previous optimization step walk
        accept_rate = sum(acceptance_rate_list)/n_walkers
        if accept_rate > .5 #.574 #settles on .4 after a bit of playing around
            σ *= 1.05
        elseif accept_rate < .5 #.574
            σ *= 0.95
        end
        #average over walkers
        delta_conj_delta=sum(delta_conj_delta_list)/n_walkers
        delta=sum(delta_list)/n_walkers
        delta_conj_el=sum(delta_conj_el_list)/n_walkers
        el=sum(el_list)/n_walkers
        El_walk=vcat(El_walk_list...)
        tl=sum(tl_list)/n_walkers
        vl=sum(vl_list)/n_walkers
        el_std_dev, error_bars = mc_stats(El_walk/ne)[[2,4]]
        ###
        f .= delta_conj_el - conj.(delta) * el #force
        S .= delta_conj_delta - delta' .* delta #quantum metric
        t_opt_hist[i]=real.(tl/ne)
        v_opt_hist[i]=real.(vl/ne)
        e_opt_hist[i]=real.(el/ne)
        e_err_opt_hist[i]=error_bars
        el_std_dev_opt_hist[i]=el_std_dev
        force_norm_opt_hist[i]=norm(f)
        params_opt_hist[i,:] .= params
        #update parameters though stochastic reconfiguration method
        tic=time()
        if i < n_opt #don't update params on last cycle so that the last set of params are sampled. 23 Jun 2025t
            if opt_method == "SR"
                stochasticreconfig!(δ,f,S,τ /(1+(ξ * i/n_opt)),ϵ,μ) #changed to modifying function. 12 Jul 2025.
                params .+= δ
            end
            if opt_J==true 
                J_params=params[1:NJ]
            end
            if opt_D==true D_params=reshape(params[(NJ+1):end],D_params_size) end
        end
        residual_opt_hist[i,:] .= δ
        toc=time()
        sr_time=toc-tic
        mc_time_avg += mc_time/n_opt
        sr_time_avg += sr_time/n_opt
        burn *= 0 #kill burn after first optimization cycle. 21 Aug 2025.
    end
    return J_params, D_params, e_opt_hist, t_opt_hist, v_opt_hist, e_err_opt_hist, el_std_dev_opt_hist, force_norm_opt_hist, params_opt_hist, mc_time_avg, sr_time_avg
end

#STOCHASTIC RECONFIGURATION

function stochasticreconfig!(δ::Vector{Float64},f::Vector{ComplexF64},S::Matrix{ComplexF64},τ::Float64,ϵ::Float64,μ::Float64)
    #S_ij = ⟨Δ*iΔj⟩-⟨Δ*i⟩⟨Δj⟩. f_i = ⟨Δ*iH⟩-⟨Δ*i⟩⟨H⟩
    #Δi = ∑_R  |R⟩⟨R| ∂_{αi}ln(Ψ_α(R))
    #update parameters
    for i in axes(S,1) S[i,i] += ϵ end #for numerical stability of inversion, add some small shift to diagonal
    @assert any(isnan, S)==false
    δ .= real.(-τ * inv(real.(S)) * (real.(f) .- (ϵ*μ/τ) .* δ)) #purely real version. I.e. S is fermi_surface metric =  Re(QGT) 06 Sep 2025
end
