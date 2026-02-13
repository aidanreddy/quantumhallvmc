function calc_f(r::Vector{Float64}, G1::Vector{Float64}, G2::Vector{Float64})::Float64
    return (2 / 3) *
           sqrt(sin(dot(r, G1) / 2)^2 + sin(dot(r, G2) / 2)^2 + sin(dot(r, G1 .+ G2) / 2)^2)
end

function calc_grad_f(
    r::Vector{Float64}, G1::Vector{Float64}, G2::Vector{Float64}, f::Float64
)::Vector{Float64}
    numerator = 
        sin(dot(r, G1)) * G1 .+ sin(dot(r, G2)) * G2 .+ sin(dot(r, G1 + G2)) * (G1 + G2)
    return (2 / 3)^2 .* numerator ./ (4 * f)
end

function calc_laplacian_f(
    r::Vector{Float64}, G1::Vector{Float64}, G2::Vector{Float64}, f::Float64
)::Float64
    gradf = calc_grad_f(r, G1, G2, f)
    return (
        (2 / 3)^2 * (
            cos(dot(r, G1)) * dot(G1, G1) +
            cos(dot(r, G2)) * dot(G2, G2) +
            cos(dot(r, G1 + G2)) * dot(G1 + G2, G1 + G2)
        ) / 4 - dot(gradf, gradf)
    ) / f
end

function calc_b3(x::Real) #from chatGPT 5/16/25, 1:30:23 PM. derivative verified against finite difference.
    absx = abs(x + 1)
    if 0 <= absx < 1
        return (1 / 6) * (4 - 6 * absx^2 + 3 * absx^3)
    elseif 1 <= absx < 2
        return (1 / 6) * (2 - absx)^3
    else
        return 0.0
    end
end

function calc_dx_b3(x::Real)
    absx = abs(x + 1)
    signx = sign(x + 1)
    if 0 < absx < 1
        return (1 / 6) * signx * (-12 * absx + 9 * absx^2)
    elseif 1 < absx < 2
        return - (1 / 2) * signx * (2 - absx)^2
    else
        return 0.0
    end
end

function calc_d2x_b3(x::Real)
    ax = abs(x + 1)
    if ax == 0
        return - 2.0  # From left/right limits: (1/6)(-12)
    elseif ax < 1
        return (1 / 6) * (18 * ax - 12)
    elseif ax < 2
        return 2 - ax
    else
        return 0.0
    end
end

function calc_s(x::Float64, pm::Vector{Float64})
    S = 0.0
    M = length(pm) - 1
    for m in 0:M
        S += pm[m + 1] * calc_b3(x * M - m)
    end
    return S
end

function calc_dx_s(x::Float64, pm::Vector{Float64})
    d_s = 0.0
    M = length(pm) - 1
    for m in 0:M
        d_s += M * pm[m + 1] * calc_dx_b3(x * M - m)
    end
    return d_s
end

function calc_d2x_s(x::Float64, pm::Vector{Float64})
    d2_s = 0.0
    M = length(pm) - 1
    for m in 0:M
        d2_s += M^2 * pm[m + 1] * calc_d2x_b3(x * M - m)
    end
    return d2_s
end

function calc_u_sin_spl(
    r::Vector{Float64},
    G1::Vector{Float64},
    G2::Vector{Float64},
    c::Vector{Float64},
    γ::Float64,
)::Float64
    f = calc_f(r, G1, G2)
    pm = vcat(c[2] - 3 * γ / length(c), c)
    return calc_s(f, pm)
end

function calc_grad_u_sin_spl(
    r::Vector{Float64},
    G1::Vector{Float64},
    G2::Vector{Float64},
    c::Vector{Float64},
    γ::Float64,
)::Vector{Float64}
    pm = vcat(c[2] - 3 * γ / length(c), c)
    f = calc_f(r, G1, G2)
    gradf = calc_grad_f(r, G1, G2, f)
    return gradf * calc_dx_s(f, pm)
end

function calc_laplacian_u_sin_spl(
    r::Vector{Float64},
    G1::Vector{Float64},
    G2::Vector{Float64},
    c::Vector{Float64},
    γ::Float64,
)::Float64
    pm = vcat(c[2] - 3 * γ / length(c), c)
    f = calc_f(r, G1, G2)
    gradf = calc_grad_f(r, G1, G2, f)
    laplacianf = calc_laplacian_f(r, G1, G2, f)
    return calc_d2x_s(f, pm) * dot(gradf, gradf) + calc_dx_s(f, pm) * laplacianf
end

function calc_grad_params_u_sin_spl(r, G1, G2, c)::Vector{Float64}
    f = calc_f(r, G1, G2)
    M = length(c)
    grad_params_u = [calc_b3(f * M - m) for m in 1:M]
    grad_params_u[2] += calc_b3(f * M) #because of cusp constraint. fixed 06 Jun 2025
    return grad_params_u
end

function calc_grad_params_grad_u_sin_spl(r, G1, G2, nJ)::Vector{Vector{Float64}}
    f = calc_f(r, G1, G2)
    gradf = calc_grad_f(r, G1, G2, f)
    M = nJ
    grad_params_gradu = [calc_dx_b3(f * M - m) * gradf * M for m in 1:M]
    grad_params_gradu[2] += calc_dx_b3(f * M) * gradf * M #because of cusp constraint
    return grad_params_gradu
end

function calc_grad_params_laplacian_u_sin_spl(r, G1, G2, nJ)::Vector{Float64}
    f = calc_f(r, G1, G2)
    gradf = calc_grad_f(r, G1, G2, f)
    laplacianf = calc_laplacian_f(r, G1, G2, f)
    M = nJ
    grad_params_laplacianu = [
        (calc_d2x_b3(f * M - m) * dot(gradf, gradf) * M^2 + calc_dx_b3(f * M - m) * laplacianf * M)
        for m in 1:M
    ]
    grad_params_laplacianu[2] += (
        calc_d2x_b3(f * M) * dot(gradf, gradf) * M^2 + calc_dx_b3(f * M) * laplacianf * M
    ) #because of cusp constraint
    return grad_params_laplacianu
end

function calc_u_row(
    idx::Int64, r_diffs_row::Vector{Vector{Float64}}, u_func
)::Array{Float64}
    ne = length(r_diffs_row)
    u_row = zeros(Float64, ne)
    for j in eachindex(r_diffs_row)
        if j == idx
            continue
        end
        u_row[j] = u_func(r_diffs_row[j])
    end
    return u_row
end

function calc_dxdy_u_row(
    idx::Int64, r_diffs_row::Vector{Vector{Float64}}, dx_u_func, dy_u_func
)::Tuple{Array{Float64}, Array{Float64}}
    ne = length(r_diffs_row)
    dx_u_row = zeros(Float64, ne)
    dy_u_row = zeros(Float64, ne)
    for j in eachindex(r_diffs_row)
        if j == idx
            continue
        end
        dx_u_row[j] = dx_u_func(r_diffs_row[j])
        dy_u_row[j] = dy_u_func(r_diffs_row[j])
    end
    return dx_u_row, dy_u_row
end

function calc_laplacian_u_row(
    idx::Int64, r_diffs_row::Vector{Vector{Float64}}, laplacian_u_func
)::Array{Float64}
    ne = length(r_diffs_row)
    laplacian_u_row = zeros(Float64, ne)
    for j in eachindex(r_diffs_row)
        if j == idx
            continue
        end
        laplacian_u_row[j] = laplacian_u_func(r_diffs_row[j])
    end
    return laplacian_u_row
end

function calc_u(r_diffs::Matrix{Vector{Float64}}, u_func)::Float64
    ne = size(r_diffs, 1)
    U_total = 0.0
    #this vectorization doesnt appear to have much of an effect on performance
    for i in 2:ne
        U_total += sum(u_func.(r_diffs[i, 1:(i - 1)]))
    end
    return U_total
end

#Jastrow kinetic
function calc_u_kin(
    r_diffs::Matrix{Vector{Float64}}, dx_u_func, dy_u_func, laplacian_u_func
)::Tuple{Matrix{ComplexF64}, Vector{ComplexF64}}
    ne = size(r_diffs)[1]
    grad_U = zeros(ComplexF64, (ne, 2))
    laplacian_U = zeros(ComplexF64, ne)
    for i in 2:ne, j in 1:(i - 1)
        gradu = [dx_u_func(r_diffs[i, j]), dy_u_func(r_diffs[i, j])]
        laplacianu = laplacian_u_func(r_diffs[i, j])
        grad_U[i, :] .+= gradu
        grad_U[j, :] .-= gradu #minus because argument is ri-rj
        laplacian_U[i] += laplacianu
        laplacian_U[j] += laplacianu
    end
    return grad_U, laplacian_U
end
