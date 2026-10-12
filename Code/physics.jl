function density_model(r, θ)
    r0 = 0.8
    rkappa = 12.0
    r_total = sqrt(r*r - 2.0 * rkappa * r * sin(θ) + rkappa*rkappa)
    return exp(-r_total/r0)
end

function calc_azimuthal_vel(r, θ, a=0.998, m=1.0)
    en = 0.232
    rkappa = 12.0
    
    x = max(r * sin(θ), 1e-5)
    
    sqrtm = sqrt(m)
    omega = (sqrtm / ((x)^(3/2) + a*sqrtm)) * (rkappa / x)^en
    return omega
end

function intensity_at_point(u, p, a=0.998, m=1.0)
    metric = p.metric
    u0 = p.u0
    vel = @views @inbounds u[vel_mask]
    pos = @views @inbounds u[pos_mask]
    
    g = metric(SVector{D,T}(pos))
    k0 = g * vel
    r, θ = @inbounds pos[2], pos[3]

    fluid_vel = calc_azimuthal_vel(r, θ, a, m)
    
    u0[begin] = 1.0
    u0[end] = fluid_vel
    normalization = u0' * g * u0
    
    if normalization >= 0.0 || !isfinite(normalization)
        return 0.0
    end
    
    u0 .= u0 .* sqrt(-1.0 / normalization)

    k0u0 = k0' * u0
    gamma = -1.0 / k0u0 

    j0 = density_model(r, θ)
    intensity = (gamma^2) * j0
    
    return isfinite(intensity) ? intensity : 0.0
end