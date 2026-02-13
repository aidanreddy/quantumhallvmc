using SpecialFunctions
using LinearAlgebra
using Interpolations

function send_to_first_cell(R,lat_to_cart,cart_to_lat)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}
    R_lat=cart_to_lat*R
    R_lat .-= round.(R_lat) #move to first cell
    shift_lat = cart_to_lat*R .- R_lat
    shift_cart = R .- lat_to_cart*R_lat
    return R_lat, lat_to_cart*R_lat, shift_lat, shift_cart
end

#disk basis function of nLL: a†^n exp(-(r/(2ℓ))²) /√(n!) 3/24/25, 8:48:15 AM
function ϕn(r_vals::Array{Float64},n::Int64)::Vector{ComplexF64}
    return (1im .*(r_vals[:,1] .- 1im*r_vals[:,2])) .^n  .* exp.(-((r_vals[:,1]).^2 .+ (r_vals[:,2]).^2)/4) ./ (sqrt(2^n) * exp(logfactorial(n)/2))
end

function ϕnAlt(z_bar_vals::Array{ComplexF64},r_square_vals::Vector{Float64},n::Int64)::Vector{ComplexF64}
    return  (z_bar_vals) .^n  .* exp.(-r_square_vals ./4) * ((1im)^n /(sqrt(2^n) * exp(logfactorial(n)/2)))
end

function calcψnInCell(r_vals::Array{Float64},n::Int64,a_vals::Vector{Vector{Int64}},a_vals_cart::Vector{Vector{Float64}})::Vector{ComplexF64}
    #rVals must be IN CELL here
    ψn=zeros(ComplexF64,size(r_vals)[1])
    numr=size(r_vals)[1]
    num_a=size(a_vals)[1]
    r_vals_big=repeat(r_vals,num_a)
    factors_big=zeros(ComplexF64,(numr*num_a))
    A=zeros(2)
    a_cart=zeros(2)
    for i in eachindex(a_vals)
        A=a_vals[i]
        a_cart=a_vals_cart[i]
        r_vals_big[1+(i-1)*numr:i*numr,1] .+= a_cart[1]
        r_vals_big[1+(i-1)*numr:i*numr,2] .+= a_cart[2]
        factors_big[1+(i-1)*numr:i*numr] .= exp.(1im*(π*A[1]*A[2] .+ (r_vals[:,1] .*a_cart[2] .- r_vals[:,2] .* a_cart[1])/2))
    end
    ψnBig=reshape(ϕn(r_vals_big,n) .* factors_big,(numr,num_a))
    ψn=vec(sum(ψnBig,dims=2))
    return ψn
end

function calcψn(r_vals::Array{Float64},n::Int64,k::Vector{Float64},a_vals::Vector{Vector{Int64}},a_vals_cart::Vector{Vector{Float64}},lat_to_cart::Array{Float64},cart_to_lat::Array{Float64})::Vector{ComplexF64}
    #shift r -> r .- rk
    rk = [k[2],-k[1]] # rk= k × ẑ
    r_valsk=copy(r_vals)
    r_valsk[:,1] .-= rk[1]
    r_valsk[:,2] .-= rk[2]
    r_valsk_in_cell = zero.(r_valsk)
    shift_lat = zero.(r_valsk)
    shift_cart = zero.(r_valsk)
    for i in axes(r_vals,1)
        r_valsk_in_cell[i,:], shift_lat[i,:], shift_cart[i,:] = send_to_first_cell(r_valsk[i,:],lat_to_cart,cart_to_lat)[2:end]
    end
    shift_factors = exp.(-1im*((r_valsk_in_cell[:,1] .* shift_cart[:,2] .- r_valsk_in_cell[:,2] .* shift_cart[:,1])./2 .+ π*shift_lat[:,1].*shift_lat[:,2]))
    ψn = shift_factors .* calcψnInCell(r_valsk_in_cell,n,a_vals,a_vals_cart) .* exp.(1im .* (r_vals[:,1] .* k[1] .+ r_vals[:,2] .* k[2]) ./ 2)
    return ψn
end

function build_interp_funcs(nmax::Int64,numx::Int64,A1::Vector{Float64},A2::Vector{Float64},a_vals::Vector{Vector{Int64}},a_vals_cart::Vector{Vector{Float64}},lat_to_cart::Matrix{Float64},cart_to_lat::Matrix{Float64})
    xmax= 0.5 
    xvals=collect(LinRange(-xmax,xmax,numx)) #in lattice coordinates
    r_vals=zeros(Float64, (numx^2 ,2))
    for i in eachindex(xvals), j in eachindex(xvals)
        r_vals[i+(j-1)*numx,:] = xvals[i]*A1+xvals[j]*A2
    end
    interp_funcs_re=Vector{Any}(undef,nmax+1)
    interp_funcs_im=Vector{Any}(undef,nmax+1)
    for n in 0:nmax
        Ψn = reshape(calcψn(r_vals,n,[0.,0.],a_vals,a_vals_cart,lat_to_cart,cart_to_lat),(numx,numx))
        interp_funcs_re[n+1]=interpolate(real.(Ψn), BSpline(Cubic()))
        interp_funcs_im[n+1]=interpolate(imag.(Ψn), BSpline(Cubic()))
    end
    return interp_funcs_re, interp_funcs_im
end


# 05/09/2025
function interp_ll_orbitals(k::Vector{Float64},r::Vector{Float64},interp_funcs_re,interp_funcs_im,grid_dim::Int64,lat_to_cart::Array{Float64},cart_to_lat::Array{Float64})::Vector{ComplexF64}
    #shift r -> r .- dk
    # for a fixed r and k, returns orbitals of all n in use
    dk = [k[2],-k[1]] # dk= k × ẑ
    rk = r-dk
    rk_c_in_cell, rk_in_cell, shift_lat, shift_cart = send_to_first_cell(rk,lat_to_cart,cart_to_lat)
    ψnRaw = zeros(ComplexF64,length(interp_funcs_re))
    rk_grid = ((rk_c_in_cell .+ 0.5) .* (grid_dim-1)) .+ 1 #the grid coordinate corresponds to an index of the grid points
    for i in eachindex(interp_funcs_re)
        ψnRaw[i] = interp_funcs_re[i](rk_grid[1],rk_grid[2]) + 1im * interp_funcs_im[i](rk_grid[1],rk_grid[2])
    end
    shift_factors = exp(-1im*((rk_in_cell[1] * shift_cart[2] - rk_in_cell[2] * shift_cart[1])/2 + π*shift_lat[1]*shift_lat[2]))
    ψn = ψnRaw * shift_factors * exp(1im * dot(r,k)/ 2)
    return ψn
end

#5/8/25, 4:46:02 PM
#this works confirmed 5/8/25, 11:00:33 PM
#need to evaluate on a grid from -0.5 to +0.5 in cell units
function calcψn_Interp(r_vals::Array{Float64},interp_func_re,interp_func_im,grid_dim::Int64,k::Vector{Float64},lat_to_cart::Array{Float64},cart_to_lat::Array{Float64})#::Vector{ComplexF64}
    #shift r -> r .- rk
    rk = [k[2],-k[1]] # rk= k × ẑ
    r_valsk=copy(r_vals)
    r_valsk[:,1] .-= rk[1]
    r_valsk[:,2] .-= rk[2]
    r_c_valsk_in_cell = zero.(r_valsk)
    r_valsk_in_cell = zero.(r_valsk)
    shift_lat = zero.(r_valsk)
    shift_cart = zero.(r_valsk)
    for i in axes(r_vals,1)
        r_c_valsk_in_cell[i,:], r_valsk_in_cell[i,:], shift_lat[i,:], shift_cart[i,:] = send_to_first_cell(r_valsk[i,:],lat_to_cart,cart_to_lat)
    end
    ψnRaw = zeros(ComplexF64,size(r_vals)[1])
    for i in axes(r_vals,1)
        rk_grid = ((r_c_valsk_in_cell[i,:] .+ 0.5) .* (grid_dim-1)) .+ 1 #the grid coordinate corresponds to an index of the grid points
        ψnRaw[i] = interp_func_re(rk_grid[1],rk_grid[2]) + 1im * interp_func_im(rk_grid[1],rk_grid[2])
    end
    shift_factors = exp.(-1im*((r_valsk_in_cell[:,1] .* shift_cart[:,2] .- r_valsk_in_cell[:,2] .* shift_cart[:,1])./2 .+ π*shift_lat[:,1].*shift_lat[:,2]))
    ψn = shift_factors .* ψnRaw .* exp.(1im .* (r_vals[:,1] .* k[1] .+ r_vals[:,2] .* k[2]) ./ 2)
    return ψn
end

#new version 3/24/25, 5:33:17 PM, vectorized
function calcr_and_factors_big(r_vals::Array{Float64},a_vals::Vector{Vector{Int64}},a_vals_cart::Vector{Vector{Float64}})::Tuple{Array{Float64},Array{ComplexF64}}
    numr=size(r_vals)[1]
    num_a=size(a_vals)[1]
    r_vals_big=repeat(r_vals,num_a)
    factors_big=zeros(ComplexF64,(numr*num_a))
    for i in eachindex(a_vals)
        A=a_vals[i]
        a_cart=a_vals_cart[i]
        for j in axes(r_vals,1)
            r_vals_big[(i-1)*numr+j,1] += a_cart[1]
            r_vals_big[(i-1)*numr+j,2] += a_cart[2]
            factors_big[(i-1)*numr+j] = exp(1im*(π*A[1]*A[2] + (r_vals[j,1] *a_cart[2] - r_vals[j,2] * a_cart[1])/2))
        end
    end
    return r_vals_big, factors_big
end

function calcψnInCellTogether(r_vals::Array{Float64},n_max::Int64,a_vals::Vector{Vector{Int64}},a_vals_cart::Vector{Vector{Float64}})::Array{ComplexF64}
    #rVals must be IN CELL here
    numr=size(r_vals)[1]
    num_a=size(a_vals)[1]
    r_vals_big, factors_big = calcr_and_factors_big(r_vals,a_vals,a_vals_cart)
    ψnBig=zeros(ComplexF64,(numr,n_max+1))
    z_bar_big=r_vals_big[:,1] .- (1im * r_vals_big[:,2])
    r_square_big= abs.(z_bar_big) .^2 #rValsBig[:,1] .^2 .+ rValsBig[:,2] .^2
    for n in 0:n_max
        ψnBig[:,n+1] = sum(reshape(ϕnAlt(z_bar_big,r_square_big,n) .* factors_big, (numr,num_a)),dims=2)
    end
    return ψnBig
end

function calcψnTogether(r_vals::Array{Float64},n_max::Int64,k::Vector{Float64},a_vals::Vector{Vector{Int64}},a_vals_cart::Vector{Vector{Float64}},lat_to_cart::Array{Float64},cart_to_lat::Array{Float64})::Array{ComplexF64}
    #shift r -> r .- rk
    rk = [k[2],-k[1]] # rk= k × ẑ
    r_valsk=copy(r_vals)
    r_valsk[:,1] .-= rk[1]
    r_valsk[:,2] .-= rk[2]
    r_valsk_in_cell = zero.(r_valsk)
    shift_lat = zero.(r_valsk)
    shift_cart = zero.(r_valsk)
    for i in axes(r_vals,1)
        r_valsk_in_cell[i,:], shift_lat[i,:], shift_cart[i,:] = send_to_first_cell(r_valsk[i,:],lat_to_cart,cart_to_lat)[2:end]
    end
    shift_factors = exp.(-1im*((r_valsk_in_cell[:,1] .* shift_cart[:,2] .- r_valsk_in_cell[:,2] .* shift_cart[:,1])./2 .+ π*shift_lat[:,1].*shift_lat[:,2]))
    ψnBigPreShift=calcψnInCellTogether(r_valsk_in_cell,n_max,a_vals,a_vals_cart)
    ψnBig=zero.(ψnBigPreShift)
    exp_factors=exp.(1im .* (r_vals[:,1] .* k[1] .+ r_vals[:,2] .* k[2]) ./ 2)
    for i in axes(ψnBig,2)
        ψnBig[:,i] =  ψnBigPreShift[:,i] .* shift_factors .* exp_factors
    end
    return ψnBig
end
