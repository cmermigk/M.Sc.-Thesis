using OrdinaryDiffEq, DiffEqCallbacks, ForwardDiff, LinearAlgebra, StaticArrays


function get_intersection(p1, p2, is_thick_disk)
    # return nothing
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    if is_thick_disk
        rmin, rmax, zmax = 8.5, 20.0, 3.0
    else
        rmin, rmax, zmax = 3.0, 6.0, 5e-3
    end

    if (abs(z2) < zmax) && (rmin < norm(p2) < rmax)
        return true
    elseif z1 * z2 > 0
        return false
    elseif z2 != z1
        t = -z1 / (z2 - z1)
        x_intersect = x1 + t * (x2 - x1)
        y_intersect = y1 + t * (y2 - y1)

        distance_to_center = norm((x_intersect, y_intersect))
        return rmin < distance_to_center < rmax
    else
        return false
    end
end

term_condition(u, λ, integrator) = true

function term_affect!(integrator)
    if any(!isfinite, integrator.u)
        integrator.p.flag = -1.0
        terminate!(integrator)
        return
    end

    r = @inbounds integrator.u[2]
    if r < 1.15
        integrator.p.flag = -1.0
        terminate!(integrator)
    elseif r > 500.0
        terminate!(integrator)
    end
end

proj_condition(u, λ, integrator) = true

function proj_affect!(integrator)
    u = integrator.u
    if any(!isfinite, u)
        integrator.p.flag = -1.0
        terminate!(integrator)
        return
    end

    dt = integrator.dt
    p = integrator.p
    g = constraints

    try
        g_prime = ForwardDiff.jacobian(x -> g(x, p), u)
        if any(!isfinite, g_prime)
            integrator.p.flag = -1.0
            terminate!(integrator)
            return
        end

        inv_g_sq = inv(g_prime * g_prime')
        
        n = size(g(u, p), 1)
        λ = zeros(n)
        Δλ = zeros(n)
        i = 1
        tol = 1e-16 #100 * dt^(p.order + 1)
        prev = zeros(n)

        while true
            if mod(i, 50) == 0
                g_prime .= ForwardDiff.jacobian(x -> g(x, p), u)
                inv_g_sq .= inv(g_prime * g_prime')
            end

            test_u = u + g_prime' * λ
            if any(!isfinite, test_u)
                integrator.p.flag = -1.0
                terminate!(integrator)
                return
            end

            g_val = g(test_u, p)
            if any(!isfinite, g_val)
                integrator.p.flag = -1.0
                terminate!(integrator)
                return
            end

            Δλ = -inv_g_sq * g_val
            λ += Δλ
            val = norm(Δλ)

            if any(!isfinite, Δλ)
                integrator.p.flag = -1.0
                terminate!(integrator)
                return
            end

            if val < max(tol, 2e-16) || i > 500
                break
            end
            i += 1
            prev .= Δλ
        end

        unew = u .+ g_prime' * λ
        if any(!isfinite, unew)
            integrator.p.flag = -1.0
            terminate!(integrator)
            return
        end

        copyto!(integrator.u, unew)
        integrator.u_modified = true

        p1 = @views @inbounds polar2cart(integrator.uprev[cart_mask])
        p2 = @views @inbounds polar2cart(integrator.u[cart_mask])
        dist = norm(p1 - p2)
        
        if get_intersection(p1, p2, p.is_thick_disk)
            integrator.p.flag += 1.0
            
            if p.is_thick_disk
                dIv = intensity_at_point(integrator.u, p)
                integrator.p.Iv += dIv * dt
            else
                terminate!(integrator) 
            end
            
        elseif @views @inbounds unew[2] < 1.15 || (dist > 5 && integrator.uprev[2] < 2)
            integrator.p.flag = -1.0
            terminate!(integrator) 
        elseif @views @inbounds unew[2] > 500.0
            terminate!(integrator) 
        end
    catch
        integrator.p.flag = -1.0
        terminate!(integrator)
        return
    end
end

function constraints(u, p)
    if any(!isfinite, u)
        return [NaN, NaN, NaN]
    end
    g = @views @inbounds p.metric(u[pos_mask])
    if any(!isfinite, g)
        return [NaN, NaN, NaN]
    end

    lagrange = @views @inbounds dot(u[vel_mask], g, u[vel_mask])

    gtt = g[1, 1]
    gtφ = g[1, 4]
    gφφ = g[4, 4]
    energy = gtt * u[D + 1] + gtφ * u[2D] - p.E
    angular_momentum = gφφ * u[2D] + gtφ * u[D + 1] - p.L
    return [lagrange, energy, angular_momentum]
end

const cb1 = DiscreteCallback(term_condition, term_affect!)
const cb2 = DiscreteCallback(proj_condition, proj_affect!)
const cb = cb2

(c::DiscreteCallback)() = c