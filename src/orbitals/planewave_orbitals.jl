function calc_planewave(k::Vector{Float64},r::Vector{Float64},RL::Vector{Vector{Float64}})::Vector{ComplexF64}
    return  cis.([dot(k .+ G,r) for G in RL])
end