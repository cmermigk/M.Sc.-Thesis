using OrdinaryDiffEq, DiffEqCallbacks
using ForwardDiff
using MutableNamedTuples
const mnt = MutableNamedTuple

function christoffel(metric::Metric, x::SVector{D,T}) where {Metric, T}
    g = metric(x)
    dg = reshape(ForwardDiff.jacobian(metric,x),(D,D,D))
    @fastmath begin

    gu = inv(g)
    Γl = @SArray T[(dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
                   for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[gu[a,1] * Γl[1,b,c] +
              gu[a,2] * Γl[2,b,c] +
              gu[a,3] * Γl[3,b,c] +
              gu[a,4] * Γl[4,b,c]
              for a in 1:D, b in 1:D, c in 1:D]
    end
end


function geodesic!(du::AbstractArray, r::AbstractArray, p, λ::T) where {T}
    x = @views @inbounds SVector{D,T}(r[pos_mask])
    u = @views @inbounds SVector{D,T}(r[vel_mask])
    Γ = christoffel(p.metric, x)

    @fastmath begin
        @inbounds for idx in 1:D
            du[idx] = r[idx+D]
            du[D+idx] = -sum(@MMatrix T[Γ[idx,b,c] * u[b] * u[c]
                                            for b in 1:D, c in 1:D])
        end
    end
        #println(du)
end

function integrate_geodesics_edu(input, metric)
    tspan = (T(0), T(1500))
    tol = 1e-9

    gtt = @views @inbounds metric(input[pos_mask])[1, 1]
    gtφ = @views @inbounds metric(input[pos_mask])[1, 4]
    gφφ = @views @inbounds metric(input[pos_mask])[4, 4]
    E = gtt * input[D + 1] + gtφ * input[2D]
    L = gφφ * input[2D] + gtφ * input[D + 1]
    
    p = mnt(metric = metric, order = 7, flag = 0.0, save = true, E = E, L = L, is_thick_disk=false)

    prob = ODEProblem(geodesic!, input, tspan, p)
    
    sims = solve(prob, Vern7(), callback = cb2,
                 abstol = tol, reltol = tol, maxiters = 3000,
                 save_everystep = true, verbose = false, dense = false, progress = false)
                 
    sol = @views @inbounds reduce(hcat, sims.u)'[:, [2, 3, 4]]
    p_final = sims.prob.p
    return (sol, p_final)
end


function integrate_geodesics_test(input,metric)
    println("Νέος κώδικας")
    tspan = (T(0),T(3500.0))
    tol=1e-15

    gtt = @views @inbounds metric(input[pos_mask])[1,1]
    gtφ = @views @inbounds metric(input[pos_mask])[1,4]
    gφφ = @views @inbounds metric(input[pos_mask])[4,4]
    E = gtt*input[D+1]+gtφ*input[2D]
    L = gφφ*input[2D]+gtφ*input[D+1]
    p = mnt(metric=metric, order = 5, flag = 0.0, save=true, E=E, L=L, is_thick_disk=false, Iv=0.0)
    prob = ODEProblem(geodesic!, input, tspan, p)

    (@isdefined cbb) ? cbb=CallbackSet(cb,cbb) : cbb=cb

    sims = solve(prob, Tsit5(), callback=cb2,
                 abstol=tol, reltol=tol, 
                 maxiters=1_000_000, 
                 verbose=false, dense=false, progress=true)
                 
    sol = @views @inbounds reduce(hcat,sims.u)'
    println("T = ",sims.t[end])
    t = sims.t
    sims = nothing
    return sol,t
end

function integrate_geodesics_scientific(input, metric, is_thick_disk::Bool=false)
    tspan = (T(0),T(3000))
    tol=1e-10

    gtt = @views @inbounds metric(input[pos_mask])[1,1]
    gtφ = @views @inbounds metric(input[pos_mask])[1,4]
    gφφ = @views @inbounds metric(input[pos_mask])[4,4]
    E = gtt*input[D+1]+gtφ*input[2D]
    L = gφφ*input[2D]+gtφ*input[D+1]
    p = mnt(
        metric = metric, 
        order = 7, 
        flag = 0.0, 
        save = false, 
        E = E, 
        L = L,
        Iv = 0.0, 
        u0 = MVector{D,T}(zeros(T,4)),
        is_thick_disk = is_thick_disk 
    )
    prob = ODEProblem(geodesic!, input, tspan, p)

    sims = solve(prob, Vern7(), callback=cb2,
     abstol=tol, reltol=tol, dense=false,
     progress = false, verbose = false)
                 
    sol = @views @inbounds sims.u[end][cart_mask]
    p_final = sims.prob.p
    sims = nothing
    return sol, p_final
end