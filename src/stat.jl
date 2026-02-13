using Statistics, FFTW

function binning_error_analysis(bin_lengths, data)
    bin_nums = Int64.(length(data) ./ bin_lengths)
    error_bars = zeros(Float64, length(bin_lengths))
    data_avg = sum(data) / length(data)
    for i in eachindex(bin_lengths)
        bin_avgs = zeros(Float64, bin_nums[i])
        for j in 1:bin_nums[i]
            bin_avgs[j] = sum(
                data[((j - 1) * bin_lengths[i] + 1):(j * bin_lengths[i])]
            ) / bin_lengths[i]
        end
        error_bars[i] = sqrt(sum((bin_avgs .- data_avg) .^ 2)) / (bin_nums[i])
    end
    return error_bars, bin_nums
end

#autocorrelation time analysis from chatGPT. 19 Aug 2025

# --- FFT-based autocorrelation function ---
function autocorrelation_fft(data::AbstractVector; norm::Bool = true)
    n = length(data)
    x = data .- mean(data)
    xzp = vcat(x, zeros(n))
    fftx = fft(xzp)
    acf_full = real(ifft(abs.(fftx) .^ 2))
    acf = acf_full[1:n] ./ (n:-1:1)
    norm ? (acf ./= acf[1]) : acf
end

# --- IMSE autocorrelation time estimator ---
function tau_int_imse(acf::AbstractVector)
    ρ = acf
    n = length(ρ)
    # pair sums Γ_k = ρ_{2k-1} + ρ_{2k}
    Γ = Float64[]
    k = 1
    while 2k <= n
        push!(Γ, ρ[2k - 1] + ρ[2k])
        k += 1
    end
    # monotone nonincreasing, nonnegative envelope
    Γmon = similar(Γ)
    curmin = Inf
    for i in eachindex(Γ)
        curmin = min(curmin, Γ[i])
        Γmon[i] = max(curmin, 0.0)
    end
    τ = 0.5 + sum(Γmon)
    return τ
end

# --- Main Monte Carlo statistics function ---
"""
    mc_stats(data)

compute mean, variance, integrated autocorrelation time (IMSE), and standard error
for a monte carlo dataset.

returns: (mean, variance, τ_Int, standard_error)
"""
function mc_stats(data::AbstractVector)
    μ = mean(data)
    σ2 = var(data)
    acf = autocorrelation_fft(data)
    τ = tau_int_imse(acf)
    σ_err = sqrt(2 * τ * σ2 / length(data))
    return μ, σ2, τ, σ_err, acf
end

#exponentially weighted moving average
function ewma(x, α)
    y = similar(x)
    y[1] = x[1]
    for i in 2:length(x)
        y[i] = α * x[i] + (1 - α) * y[i - 1]
    end
    return y
end
