module blast_tutorials

using Tullio
using DataInterpolations
using Interpolations
using LinearAlgebra

export get_nmodes_fullsky, compute_kernels, Σ

function get_nmodes_fullsky(ls)
    """ Returns the number of modes in each ell bin """
    nmodes = [ls[i+1]^2 - ls[i]^2 for i in 1:length(ls)-1]
    lp = ls[end]^2 / ls[end-1]
    push!(nmodes, lp^2 - ls[end]^2)
    return 0.5 * nmodes
end

function make_grid(χ, R)
    return vec(χ * R')
end

function grid_interpolator(W, χ, grid)

    W_interp = zeros(length(W[:,1]), length(grid))

    for i in 1:length(W[:,1])
        interp = AkimaInterpolation(W[i,:], χ, extrapolate=true)
        W_interp[i,:] = interp.(grid)
    end

    return W_interp
end

function grid_interpolator(W, grid, label::String)
    if label == "C"
        W_array = W["kernels_cl"]
    elseif label == "L"
        W_array = W["kernels_sh"]
    else
        error("Label must be C or L!!!!!!!")
    end

    χ = W["chi_sh"]

    return grid_interpolator(W_array, χ, grid)
end

function compute_kernels(W, χ, R)

    nχ = length(χ)
    nR = length(R)

    W_C = reshape(grid_interpolator(W, make_grid(χ, R), "C"), 10, nχ, nR)

    χ2_app = zeros(5, nχ*nR)
    for i in 1:5
        χ2_app[i,:] = make_grid(χ, R) .^ 2
    end

    W_L = grid_interpolator(W, make_grid(χ, R), "L")
    W_L = reshape( W_L./χ2_app , 5, nχ, nR)

    W_C_r1 = W_C[:,:,end]
    W_L_r1 = W_L[:,:,end]

    @tullio K_CC[i,j,c,r] := W_C_r1[i,c] * W_C[j,c,r] + W_C[i,c,r]*W_C_r1[j,c]

    @tullio K_LL[i,j,c,r] := W_L_r1[i,c] * W_L[j,c,r] + W_L[i,c,r]*W_L_r1[j,c]

    @tullio K_CL[i,j,c,r] := W_C_r1[i,c] * W_L[j,c,r] + W_C[i,c,r]*W_L_r1[j,c]

    return K_CC, K_CL, K_LL
end

function Σ(Cℓ_CC, Cℓ_LL, Cℓ_CL, ℓ,dtype,f_sky=1)
    n̄_C = dtype(4 * (3437.746771)^2)
    n̄_L = dtype(27/5 * (3437.746771)^2)
    σ_ϵ = dtype(0.28)

    N_CC = dtype.(1/n̄_C .* I(10))
    N_LL = dtype.(σ_ϵ^2/n̄_L .* I(5))

    @tullio Σ_CC[l,i,j] := sqrt(2/(f_sky*(2*ℓ[l]+1))) * (Cℓ_CC[l,i,j] + N_CC[i,j])
    @tullio Σ_LL[l,i,j] := sqrt(2/(f_sky*(2*ℓ[l]+1))) * (Cℓ_LL[l,i,j] + N_LL[i,j])
    @tullio Σ_CL[l,i,j] := sqrt(2/(f_sky*(2*ℓ[l]+1))) * Cℓ_CL[l,i,j]

    return dtype.(Σ_CC), dtype.(Σ_LL), dtype.(Σ_CL)
end

function Cℓ_limber_nl(power_spectrum_nl_limber, ℓ, χ, tracers, width)
    n = length(χ)
    stringname = "../N5K/input/kernels_" * width * "width.npz"
    W = npzread(stringname)
    WA = W["kernels_sh"]
    WB = W["kernels_cl"]
    WA_interp = zeros(5,n)
    WB_interp = zeros(10,n)

    for i in 1:5
        interp = BSplineInterpolation(WA[i,:], W["chi_sh"], 3, :ArcLen, :Average, extrapolate=true)
        WA_interp[i,:] = interp.(χ)
    end

    for i in 1:10
        interp = BSplineInterpolation(WB[i,:], W["chi_cl"], 3, :ArcLen, :Average, extrapolate=true)
        WB_interp[i,:] = interp.(χ)
    end

    if tracers == "CC"
        F = 1
        KA = WB_interp
        KB = WB_interp
    elseif tracers == "CL"
        F = sqrt.(factorial_frac(ℓ))*(ℓ+0.5)^(-2)
        KA = WB_interp
        KB = WA_interp
    elseif tracers == "LL"
        F = factorial_frac(ℓ)*(ℓ+0.5)^(-4)
        KA = WA_interp
        KB = WA_interp
    end

    Δχ = ((χ[n]-χ[1])/(n-1))
    pesi = SimpsonWeightArray(n)

    pk_over_chi = power_spectrum_nl_limber(ℓ, χ) ./ (χ .^ 2)

    @tullio Cℓ[i,j] := Δχ*pk_over_chi[m]*KA[i,m]*KB[j,m]*pesi[m]
    return Cℓ
end

function Cℓ_limber(power_spectrum_limber, ℓ, χ, tracers, width)
    n = length(χ)
    stringname = "../N5K/input/kernels_" * width * "width.npz"
    W = npzread(stringname)
    WA = W["kernels_sh"]
    WB = W["kernels_cl"]
    WA_interp = zeros(5,n)
    WB_interp = zeros(10,n)

    for i in 1:5
        interp = BSplineInterpolation(WA[i,:], W["chi_sh"], 3, :ArcLen, :Average, extrapolate=true)
        WA_interp[i,:] = interp.(χ)
    end

    for i in 1:10
        interp = BSplineInterpolation(WB[i,:], W["chi_cl"], 3, :ArcLen, :Average, extrapolate=true)
        WB_interp[i,:] = interp.(χ)
    end

    if tracers == "CC"
        F = 1
        KA = WB_interp
        KB = WB_interp
    elseif tracers == "CL"
        F = sqrt.(factorial_frac(ℓ))*(ℓ+0.5)^(-2)
        KA = WB_interp
        KB = WA_interp
    elseif tracers == "LL"
        F = factorial_frac(ℓ)*(ℓ+0.5)^(-4)
        KA = WA_interp
        KB = WA_interp
    end

    Δχ = ((χ[n]-χ[1])/(n-1))
    pesi = SimpsonWeightArray(n)

    pk_over_chi = power_spectrum_limber(ℓ, χ) ./ (χ .^ 2)

    @tullio Cℓ[i,j] := Δχ*pk_over_chi[m]*KA[i,m]*KB[j,m]*pesi[m]
    return Cℓ
end

end 
