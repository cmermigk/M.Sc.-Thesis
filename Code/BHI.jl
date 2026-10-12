module BHI

using ProgressMeter
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Images, ImageView
using LaTeXStrings
using GLMakie, GeometryBasics, Rotations
using StaticArrays, LinearAlgebra
using OhMyThreads
using Colors, ColorSchemes

const D = 4
global const T = Float64
const pos_mask = [true, true, true, true, false, false, false, false]
# const tmp_mask = .~pos_mask
# tmp_mask[end] = false
const vel_mask = .~pos_mask #copy(tmp_mask)
const cart_mask = [false, true, true, true,false, false, false, false]
global const prog = Ref{Progress}(Progress(1))

include("callbacks.jl")
include("initial_conditions.jl")
include("educational.jl")
include("metrics.jl")
include("integrator.jl")
include("physics.jl")
include("shadow.jl")

const background = Images.load(joinpath(@__DIR__, "background_images", "test_image.jpg"))#milky_way Τ(30.0) alims, blims -12,12
const h, w = size(background)
# const init_r = T(500.0)
const init_r = T(30.0) 

function cart2polar(x::AbstractVector{T})::SVector{2,T} where T<:Real
    r = norm(x)
    θ = acos(x[3] / r)  
    φ = atan(x[2], x[1])
    SVector{2,T}(vcat(θ,φ))
end

function polar2cart(xx)
    r,θ,φ = xx
    x = r * sin(θ) * cos(φ)
    y = r * sin(θ) * sin(φ)
    z = r * cos(θ)
    @SVector [x,y,z]
end

function get_rotation_matrix(roll::Number,pitch::Number,yaw::Number)::AbstractArray

    R_euler = RotZYX(deg2rad(yaw),
                     deg2rad(pitch),
                     deg2rad(roll))

    R_euler
end

function get_background_color(angs, background=nothing)
    θ, φ = angs
    
 
    u = 1.0 - mod((φ - π) / π, 1.0)
    v = mod(θ, π) / π

    px = clamp(round(Int, u * w), 1, w)
    py = clamp(round(Int, v * h), 1, h)

    return @inbounds background[py, px]
end

function trace_rays_edu(metric::Metric, sc::Int64) where Metric
    dist = T(1)
    N, M = sc, sc
    println("EDU Grid: ", M, " x ", N)

    # Ίδια ακριβώς όρια (FOV) με το sci
    alims = (-9, 9)
    blims = (-9, 9)
    # alims = (-12, 12)
    # blims = (-12, 12)
    # alims = (-20, 20) 
    # blims = (-20, 20)
    r_cam = init_r 
    θ0 = deg2rad(87.5)
    pos = SVector{D,T}([0, r_cam, θ0, π/2]) 
    
    inputs = generate_rays(pos, metric, N, M, alims, blims, dist)

    prog[] = Progress(N * M)

    sols = tmap(1:(N*M)) do i
        sol, p_end = integrate_geodesics_edu(inputs[i], metric)
        next!(prog[])
        return (sol, p_end)
    end

    fig = Figure(size=(1200, 600))
    
    ax = LScene(fig[1,1], show_axis=false,
                scenekw = (
                    lights = [DirectionalLight(RGB{N0f8}(0.3,0.3,0.3), Point3f([-r_cam-0.5, 0, 0])),
                              AmbientLight(colorant"white")],
                    ))
    
    mesh!(ax, Sphere(Point3f(0), 1.0f0), color = :black)
    plot_accretion_disk(ax)
    mesh!(ax, Sphere(Point3f(polar2cart(pos[2:4])), 5.0f0), color = :purple)

    img = map(sols) do sims_tuple
        sims = sims_tuple[1]
        flag = sims_tuple[2].flag
        
        pos_final = @views sims[end, :]
        angs = pos_final[2:3]

        if flag > 0.0
            return RGB{N0f8}(1.0, 1.0, 1.0)
            # return get_background_color(angs, background)
        elseif flag < 0.0
            return RGB{N0f8}(0.0, 0.0, 0.0)
            # return get_background_color(angs, background)
        else
            return get_background_color(angs, background)
        end
    end

    img_2d = reshape(img, M, N)

    n_traj = N * M
    n_shown = 150
    indices = n_traj <= 150 ? (1:n_traj) : rand(1:n_traj, n_shown)

    rmax = 0.0
    @showprogress for i in indices
        sol_cart = reduce(hcat, map(polar2cart, eachrow(sols[i][1])))'
        lines!(ax, sol_cart[:,1], sol_cart[:,2], sol_cart[:,3], color=img[i])
        
        r = @views @inbounds norm(sol_cart[end, :])
        if r > rmax
            rmax = r
        end
    end

    plot_celestial_sphere(ax, max(rmax, r_cam), I, background)
    ax2 = GLMakie.Axis(fig[1,2])#, xlabel="β/M", ylabel="α/M", aspect=DataAspect())
    hidedecorations!(ax2)
    image!(ax2, blims, alims, img_2d)

    display(fig)
    return fig
end

function trace_rays_scientific(metric::Metric, sc::Int64) where Metric

    dist = T(1)
    N = sc
    M = sc
    println(N," ",M)

    θ0 = deg2rad(87.5)
    # alims = (-9,9)
    # blims = (-9,9)
    alims = (-20, 20)
    blims = (-20, 20)
    # alims = (-30,30)
    # blims = (-60,60)

    pos = SVector{D,T}([0,init_r,θ0,π/2])
    inputs = generate_rays(pos, metric, N, M, alims, blims, dist)

    prog[] = Progress(N*M)

    sols = tmap(inputs) do input
        sol, p = integrate_geodesics_scientific(input, metric)
        next!(prog[])
        return (sol, p)
    end

    img = map(reshape(sols,N,M)) do sims
        pos_final = @views @inbounds sims[1]
        flag = sims[2].flag
        angs = @views @inbounds pos_final[2:3]

        if flag > 0.0
             return RGB{N0f8}(1.0, 1.0, 1.0)
            #  return get_background_color(angs, background)
        elseif flag < 0.0
             return RGB{N0f8}(0.0, 0.0, 0.0)
            # return get_background_color(angs, background)
        else
             return get_background_color(angs, background)
        end
    end

    img_2d = reshape(img, M, N)

    fig = Figure(backgroundcolor=:black, size=(M, N), figure_padding=0)
    
    ax = GLMakie.Axis(fig[1,1], backgroundcolor=:black, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false

    image!(ax, blims, alims, img_2d)

    display(fig)

    filename = "trace_rad_$(N)x$(M).png"
    save(filename, fig; size=(M, N))
    
    println("Η εικόνα γεωμετρίας αποθηκεύτηκε επιτυχώς ως: ", filename)
end

function trace_rays(metric, sc, flag="sci")
    if flag == "edu"
        trace_rays_edu(metric, sc)
    elseif flag == "sci"
        trace_rays_scientific(metric, sc)
    elseif flag == "shadow"
        trace_rays_shadow(metric, sc, false)
    elseif flag == "shadow_bg"
        trace_rays_shadow(metric, sc, true)
    else
        throw(ArgumentError("Not supported method: $flag"))
    end
end

function energy(metric, x)
    pos = @views @inbounds x[pos_mask]
    vel = @views @inbounds x[vel_mask]
    g = metric(pos)
    gtt = @views @inbounds g[1, 1]
    gtφ = @views @inbounds g[1, 4]
    gφφ = @views @inbounds g[4, 4]

    E = @inbounds gtt * vel[1] + gtφ * vel[end]
    return E
end

function ang_momentum(metric, x)
    pos = @views @inbounds x[pos_mask]
    vel = @views @inbounds x[vel_mask]
    g = metric(pos)
    gtt = @views @inbounds g[1, 1]
    gtφ = @views @inbounds g[1, 4]
    gφφ = @views @inbounds g[4, 4]

    L = @inbounds gφφ * vel[end] + gtφ * vel[1]
    return L
end

function lagrange(metric, x)
    pos = @views @inbounds x[pos_mask]
    vel = @views @inbounds x[vel_mask]
    g = metric(pos)
    return vel' * g * vel
end

function get_final_velocity(sims)
    return sims.u[end][vel_mask]
end

function get_final_velocity(rd, g, gu)
    return normalization(rd, g, gu)
end

function robustness()
    metric = kerr_metric_bl
    M = 1
    a = 0.998
    theta_0 = deg2rad(85) 

    r_ph = 3.0 

    xi_c = (M*(r_ph^2 - a^2) - r_ph*(r_ph^2 - 3*M*r_ph)) / (a*(r_ph - M))
    eta_c = (r_ph^3 * (4*a^2*M - r_ph*(r_ph - 3*M)^2)) / (a^2 * (r_ph - M)^2)

    alpha_c = -xi_c / sin(theta_0)
    beta_c  = sqrt(eta_c + a^2*cos(theta_0)^2 - xi_c^2*cot(theta_0)^2)

    alpha_direct, beta_direct     = 0.8 * alpha_c, 0.8 * beta_c      
    alpha_spiral, beta_spiral     = (1 + 1e-6) * alpha_c, beta_c      
    alpha_scatter, beta_scatter   = 1.5 * alpha_c, 1.5 * beta_c       

    list = [[beta_direct,alpha_direct,"direct"],[beta_scatter,alpha_scatter,"scatter"],[beta_spiral,alpha,"spiral"]]
    
    for item in list
        println("$(item[3])")
    pos = @SVector [0.0, r_ph, theta_0, 1.0]
    g = metric(pos)
    gu = inv(g)

    rd = gu * [0.0, -1.0, item[1], item[2]]
    rd = SVector{D,T}(vcat(0, rd[[2,3,4]]))
    ray_direction = get_final_velocity(rd, g, gu)
    println(ray_direction' * g * ray_direction)
    ray = MVector{2D,T}(vcat(pos, ray_direction))
    tspan = (0.0, 1000.0)
    # tol = 1e-12
    p = mnt(metric = metric, order = 7, flag = 0.0, save = true, E = energy(metric, ray), L = ang_momentum(metric, ray), is_thick_disk=false)
    prob = ODEProblem(geodesic!, ray, tspan, p)
    sims = solve(prob, Vern7(), callback=cb2, abstol=1e-15, reltol=1e-15, dtmax=0.001, maxiters=100000)    
    sol = reduce(hcat, sims.u)'
    λ = sims.t
    pl = reduce(hcat, map(eachrow(sol)) do x
                    @views @inbounds polar2cart(x[cart_mask])
                end)'
    N = size(sol, 1)
    pow = round(Int, log10(N))

    fig = Figure(fontsize=26, size=(1920, 1080), pt_per_unit=1, px_per_unit=1)
    set_theme!(GLMakie.theme_latexfonts(),fontsize=26)
    
    ax1 = Axis3(fig[1, 3])
    mesh!(ax1, Sphere(Point3f(0), 1.15f0), color = :black)
    xlims!(ax1, -5, 5)
    ylims!(ax1, -5, 5)
    zlims!(ax1, -5, 5)

    stepstring = latexstring(L"{\mathrm{steps}} \cdot 10^{"[1:end-1] * string(pow) * "}\$")
    
    ax6 = GLMakie.Axis(fig[1, 2], xlabel="λ", ylabel="r")

    ax2 = GLMakie.Axis(fig[1, 1], ylabel=L"\Delta r/r_0", xlabel=stepstring, yscale=log10)
    ax22 = GLMakie.Axis(fig[1, 1], xlabel="λ", yscale=log10, xaxisposition=:top)
    hidespines!(ax22)
    hideydecorations!(ax22)

    ax3 = GLMakie.Axis(fig[2, 1], ylabel=L"\Delta e/e_0", xlabel=stepstring, yscale=log10)
    
    ax4 = GLMakie.Axis(fig[2, 2], ylabel=L"\Delta l/l_0", xlabel=stepstring, yscale=log10)
    
    ax5 = GLMakie.Axis(fig[2, 3], ylabel=L"g_{\mu \nu} u^\mu u^\nu", xlabel=stepstring)

    r = sol[1, 2]
    Δr = @views @inbounds abs.(sol[2:end, 2] .- r)
    mask = Δr .== 0.0
    Δr[mask] .= 1e-16
    nsteps = 1:N

    Δe = map(eachrow(sol)) do x
             energy(metric, x)
         end
    Δl = map(eachrow(sol)) do x
        ang_momentum(metric, x)
    end
    Δn = map(eachrow(sol)) do x
        lagrange(metric, x)
    end
    Δe = abs.(diff(Δe) ./ Δe[1])
    mask = Δe .== 0.0
    Δe[mask] .= 1e-16
    Δl = abs.(diff(Δl) ./ Δl[1])
    mask = Δl .== 0.0
    Δl[mask] .= 1e-16

    lines!(ax1, pl[1:2:end, 1], pl[1:2:end, 2], pl[1:2:end, 3])
    scatterlines!(ax6, λ, sol[:, 2])

    lines!(ax2, nsteps[1:end-1] ./ (10^pow), Δr)
    p = lines!(ax22, λ[1:end-1], Δr, alpha=0.001)
    delete!(ax22, p)
    
    lines!(ax3, nsteps[1:end-1] ./ (10^pow), Δe)
    lines!(ax4, nsteps[1:end-1] ./ (10^pow), Δl)
    lines!(ax5, nsteps ./ (10^pow), Δn)
    
    display(fig)
    save("errors_$(item[3]).png", fig)
    println("Το robustness αποθηκεύτηκε επιτυχώς ως errors.png")
end
end

function plot_scattering_robustness(metric::Metric) where Metric
    println("Υπολογισμός τροχιάς σκέδασης (Scattering Orbit)...")
    
    b_param = 5.35 
    pos = SVector{D,T}([0, 30.0, deg2rad(90), 0.0])
    
    g = metric(pos)
    gu = inv(g)
    
    ray_direction = gu * [0.0, -1.0, 0.0, -b_param] 
    ray_direction = @views @inbounds SVector{D,T}(vcat(0, ray_direction[2:4]))
    
    input = collect(vcat(pos, normalization(ray_direction, g, gu)))
    
    tspan = (T(0), T(1500))
    tol = 1e-10

    gtt = @views @inbounds metric(input[pos_mask])[1, 1]
    gtφ = @views @inbounds metric(input[pos_mask])[1, 4]
    gφφ = @views @inbounds metric(input[pos_mask])[4, 4]
    
    E0 = gtt * input[D + 1] + gtφ * input[2D]
    L0 = gφφ * input[2D] + gtφ * input[D + 1]
    
    p = mnt(metric = metric, order = 7, flag = 0.0, save = true, E = E0, L = L0, is_thick_disk=false)

    prob = ODEProblem(geodesic!, input, tspan, p)
    sims = solve(prob, Vern7(), callback = cb1, 
                 abstol = tol, reltol = tol, maxiters = 10000,
                 save_everystep = true, verbose = false, dense = false)
                 
    times = sims.t
    states = sims.u
    
    ΔE = zeros(length(states))
    ΔL = zeros(length(states))
    Constraints = zeros(length(states))
    r_vals = zeros(length(states))
    
    for (i, state) in enumerate(states)
        pos_current = state[pos_mask]
        vel_current = state[vel_mask]
        g_current = metric(pos_current)
        
        E_current = g_current[1,1]*vel_current[1] + g_current[1,4]*vel_current[4]
        L_current = g_current[4,4]*vel_current[4] + g_current[1,4]*vel_current[1]
        
        ΔE[i] = abs(E_current - E0) / (abs(E0) + 1e-16)
        ΔL[i] = abs(L_current - L0) / (abs(L0) + 1e-16)
        Constraints[i] = abs(dot(vel_current, g_current, vel_current))
        r_vals[i] = pos_current[2]
    end
    
    fig = Figure(size=(1800, 900), fontsize=16)
    safe_log(x) = max.(x, 1e-16)
    
    ax_r_lam = GLMakie.Axis(fig[1, 1], title="Radial Distance (r vs λ)", ylabel="r / M")
    lines!(ax_r_lam, times, r_vals, color=:purple, linewidth=2)
    
    ax_e = GLMakie.Axis(fig[1, 2], title="Energy Conservation Error (ΔE / E₀)", yscale=log10, ylabel="Relative Error")
    lines!(ax_e, times, safe_log(ΔE), color=:blue, linewidth=1.5)
    
    ax_traj3d = LScene(fig[1, 3], show_axis=true, scenekw=(title="3D Scattering Orbit",))
    mesh!(ax_traj3d, Sphere(Point3f(0), 1.0f0), color = :black)
    cart_coords = reduce(hcat, map(polar2cart, [s[pos_mask] for s in states]))'
    lines!(ax_traj3d, cart_coords[:,1], cart_coords[:,2], cart_coords[:,3], color=:orange, linewidth=2)

    ax_l = GLMakie.Axis(fig[2, 1], title="Angular Momentum Error (ΔL / L₀)", yscale=log10, xlabel="Affine Parameter (λ)", ylabel="Relative Error")
    lines!(ax_l, times, safe_log(ΔL), color=:red, linewidth=1.5)
    
    ax_c = GLMakie.Axis(fig[2, 2], title="Null Geodesic Constraint (|g_μν u^μ u^ν|)", yscale=log10, xlabel="Affine Parameter (λ)", ylabel="Absolute Error")
    lines!(ax_c, times, safe_log(Constraints), color=:green, linewidth=1.5)

    display(fig)
    return fig
end

function scattering_from_far()
    metric = kerr_metric_bl
    M = 1
    a = 0.998
    θ₀ = deg2rad(85)  # Observer inclination
    r_start = 50.0     # Starting radius (far from BH)
 
    r_ph = 3.0  
    
    ξ_c = (r_ph^2 * (3*M - r_ph) - a^2 * (r_ph + M)) / (a * (r_ph - M))
    η_c = (r_ph^3 * (4*a^2*M - r_ph*(r_ph - 3*M)^2)) / (a^2 * (r_ph - M)^2)
    
    α_c = -ξ_c / sin(θ₀)
    β_c = sqrt(η_c + a^2*cos(θ₀)^2 - ξ_c^2*cot(θ₀)^2)
    
    scenarios = [
        (0.8 * α_c, 0.8 * β_c, "captured"),     
        (1.5 * α_c, 1.5 * β_c, "scattered"),    
        (1.0001 * α_c, β_c, "orbiting"),        
    ]
    
    for (α, β, label) in scenarios
        println("Simulating: $label")
        
        ξ = -α * sin(θ₀)
        η = β^2 - a^2*cos(θ₀)^2 + ξ^2*cot(θ₀)^2
        
        t₀ = 0.0
        r₀ = r_start
        θ₀_pos = θ₀
        φ₀ = 0.0
        
        E = 1.0
        L_z = ξ  # Conserved angular momentum
        Q = η    # Carter constant
        
        p_t = -E
        p_φ = L_z
        
        p_θ² = Q - L_z^2 * cot(θ₀_pos)^2 + a^2 * cos(θ₀_pos)^2
        p_θ = sqrt(max(p_θ², 0.0))  # Ensure non-negative
        
        pos = @SVector [t₀, r₀, θ₀_pos, φ₀]
        g = metric(pos)
        g_inv = inv(g)
        
        A = g_inv[2, 2]  # g^{rr}
        B = 0.0          # g^{rt} = 0 in BL coordinates
        C = g_inv[1,1]*p_t^2 + g_inv[3,3]*p_θ^2 + g_inv[4,4]*p_φ^2 + 2*g_inv[1,4]*p_t*p_φ
        
        p_r = -sqrt(max(-C/A, 0.0))
        
        p_cov = SVector{4,Float64}(p_t, p_r, p_θ, p_φ)
        
        u_contra = g_inv * p_cov
        
        norm_check = u_contra' * g * u_contra
        println("Null constraint check: g_μν u^μ u^ν = $norm_check")
        
        ray = MVector{8,Float64}(vcat(pos, u_contra))
        
        tspan = (0.0, 5000.0)  # Long enough for multiple orbits
        
        p_params = mnt(
            metric = metric, 
            order = 7, 
            flag = 0.0, 
            save = true, 
            E = energy(metric, ray), 
            L = ang_momentum(metric, ray), 
            is_thick_disk = false
        )
        
        prob = ODEProblem(geodesic!, ray, tspan, p_params)
        
        sims = solve(
            prob, 
            Vern7(), 
            abstol = 1e-13, 
            reltol = 1e-13, 
            dtmax = 0.1,
            maxiters = 500000
        )
        
        sol = reduce(hcat, sims.u)'
        λ = sims.t
        
        println("Integration complete: $(size(sol, 1)) points")
        println("Final radius: $(sol[end, 2])")
        
        pl = reduce(hcat, map(eachrow(sol)) do x
            @views @inbounds polar2cart(x[2:4])  # r, θ, φ → x, y, z
        end)'
        
        fig = Figure(fontsize = 24, size = (1200, 800))
        ax = Axis3(fig[1, 1], 
                   xlabel = "x", ylabel = "y", zlabel = "z",
                   title = "Trajectory: $label")
        
        r₊ = M + sqrt(M^2 - a^2)
        mesh!(ax, Sphere(Point3f(0), Float32(r₊)), color = :black)
        
        lines!(ax, pl[:, 1], pl[:, 2], pl[:, 3], color = :blue, linewidth = 2)
        
        scatter!(ax, [pl[1, 1]], [pl[1, 2]], [pl[1, 3]], color = :green, markersize = 10)
        scatter!(ax, [pl[end, 1]], [pl[end, 2]], [pl[end, 3]], color = :red, markersize = 10)
        
        xlims!(ax, -60, 60)
        ylims!(ax, -60, 60)
        zlims!(ax, -60, 60)
        
        save("trajectory_$label.png", fig)
        println("Saved: trajectory_$label.png")
    end
end

function a()
    metric = kerr_metric_bl
    a = 0.998
    M = 1.0
    
    r_c = 2.5 
    θ0 = π/2  # Εκκίνηση από τον ισημερινό
    pos = @SVector [0.0, r_c, θ0, 0.0]
    
    g = metric(pos)
    gu = inv(g)

    Φ = -(r_c^3 - 3*M*r_c^2 + a^2*r_c + M*a^2) / (a*(r_c - M))
    η = (r_c^3 * (4*M*a^2 - r_c*(r_c - 3*M)^2)) / (a^2*(r_c - M)^2)
    
    p_t = -1.0
    p_r = 0.0   # ΑΥΣΤΗΡΑ ΜΗΔΕΝ για να μην ξεφύγει από την ακτίνα r_c!
    p_θ = sqrt(abs(η + a^2 * cos(θ0)^2 - Φ^2 * cot(θ0)^2))
    p_φ = Φ
    
    p_cov = @SVector [p_t, p_r, p_θ, p_φ]
    
    ray_direction = gu * p_cov
    ray = MVector{2D,T}(vcat(pos, ray_direction))
    
    tspan = (0.0, 300.0)
    
    p = mnt(
        metric = metric, order = 7, flag = 0.0, save = true, 
        E = energy(metric, ray), L = ang_momentum(metric, ray), 
        is_thick_disk = true, Iv = 0.0, u0 = MVector{D,T}(zeros(T,4))
    )
    
    prob = ODEProblem(geodesic!, ray, tspan, p)
    sims = solve(prob, Vern7(), callback=cb2, abstol=1e-13, reltol=1e-13, maxiters=2_000_000, progress=true)
    sol = reduce(hcat, sims.u)'
    λ = sims.t
    pl = reduce(hcat, map(eachrow(sol)) do x
                    @views @inbounds polar2cart(x[cart_mask])
                end)'
    N = size(sol, 1)
    pow = round(Int, log10(N))
    
    with_theme(Makie.theme_latexfonts(), fontsize=36) do 
    fig = Figure(size=(1920, 1080), pt_per_unit=1, px_per_unit=1)
    # set_theme!(GLMakie.theme_latexfonts())
    
    ax1 = Axis3(fig[1, 3])
    mesh!(ax1, Sphere(Point3f(0), 1.15f0), color = :black)
    xlims!(ax1, -5, 5)
    ylims!(ax1, -5, 5)
    zlims!(ax1, -5, 5)

    stepstring = latexstring(L"{\mathrm{steps}} \cdot 10^{"[1:end-1] * string(pow) * "}\$")
    
    ax6 = GLMakie.Axis(fig[1, 2], xlabel="λ", ylabel="r")

    ax2 = GLMakie.Axis(fig[1, 1], ylabel=L"\Delta r/r_0", xlabel=stepstring, yscale=log10)
    ax22 = GLMakie.Axis(fig[1, 1], xlabel="λ", yscale=log10, xaxisposition=:top)
    hidespines!(ax22)
    hideydecorations!(ax22)

    ax3 = GLMakie.Axis(fig[2, 1], ylabel=L"\Delta e/e_0", xlabel=stepstring, yscale=log10)
    
    ax4 = GLMakie.Axis(fig[2, 2], ylabel=L"\Delta l/l_0", xlabel=stepstring, yscale=log10)
    
    ax5 = GLMakie.Axis(fig[2, 3], ylabel=L"g_{\mu \nu} u^\mu u^\nu", xlabel=stepstring)

    r = sol[1, 2]
    Δr = @views @inbounds abs.(sol[2:end, 2] .- r)
    mask = Δr .== 0.0
    Δr[mask] .= 1e-16
    nsteps = 1:N

    Δe = map(eachrow(sol)) do x
             energy(metric, x)
         end
    Δl = map(eachrow(sol)) do x
        ang_momentum(metric, x)
    end
    Δn = map(eachrow(sol)) do x
        lagrange(metric, x)
    end
    Δe = abs.(diff(Δe) ./ Δe[1])
    mask = Δe .== 0.0
    Δe[mask] .= 1e-16
    Δl = abs.(diff(Δl) ./ Δl[1])
    mask = Δl .== 0.0
    Δl[mask] .= 1e-16

    lines!(ax1, pl[1:2:end, 1], pl[1:2:end, 2], pl[1:2:end, 3])
    scatterlines!(ax6, λ, sol[:, 2])
    lines!(ax2, nsteps[1:end-1] ./ (10^pow), Δr)
    p = lines!(ax22, λ[1:end-1], Δr, alpha=0.001)
    delete!(ax22, p)
    
    lines!(ax3, nsteps[1:end-1] ./ (10^pow), Δe)
    lines!(ax4, nsteps[1:end-1] ./ (10^pow), Δl)
    lines!(ax5, nsteps ./ (10^pow), Δn)
    
    display(fig)
    save("errors_stable.png", fig)
    end
    println("Το robustness αποθηκεύτηκε επιτυχώς ως errors.png")
end

function b()
    metric = kerr_metric_bl
    
    pos = @SVector [0.0, 50.0, deg2rad(75), 1.0]    
    g = metric(pos)
    gu = inv(g)

    b_param = -2
    
    rd = gu * [0.0, -1.0, 0.0, b_param]
    rd = SVector{D,T}(vcat(0, rd[2:4]))
    
    ray_direction = get_final_velocity(rd, g, gu)
    ray = MVector{2D,T}(vcat(pos, ray_direction))
    
    tspan = (0.0, 1500.0)
    
    p = mnt(metric = metric, order = 7, flag = 0.0, save = true, 
            E = energy(metric, ray), L = ang_momentum(metric, ray), 
            is_thick_disk = true, Iv = 0.0, u0 = MVector{D,T}(zeros(T,4)))
    
    prob = ODEProblem(geodesic!, ray, tspan, p)
    
    sims = solve(prob, Vern7(), callback=cb2, abstol=1e-15, reltol=1e-15, maxiters=1000000, progress=true)    
    
    sol = reduce(hcat, sims.u)'
    λ = sims.t
    pl = reduce(hcat, map(eachrow(sol)) do x
                    @views @inbounds polar2cart(x[cart_mask])
                end)'
    N = size(sol, 1)
    pow = round(Int, log10(N))

        with_theme(Makie.theme_latexfonts(), fontsize=36) do 
    fig = Figure(size=(1920, 1080), pt_per_unit=1, px_per_unit=1)
    # set_theme!(GLMakie.theme_latexfonts())
    
    ax1 = Axis3(fig[1, 3])
    mesh!(ax1, Sphere(Point3f(0), 1.15f0), color = :black)
    xlims!(ax1, -5, 5)
    ylims!(ax1, -5, 5)
    zlims!(ax1, -5, 5)

    stepstring = latexstring(L"{\mathrm{steps}} \cdot 10^{"[1:end-1] * string(pow) * "}\$")
    
    ax6 = GLMakie.Axis(fig[1, 2], xlabel="λ", ylabel="r")

    ax2 = GLMakie.Axis(fig[1, 1], ylabel=L"\Delta r/r_0", xlabel=stepstring, yscale=log10)
    ax22 = GLMakie.Axis(fig[1, 1], xlabel="λ", yscale=log10, xaxisposition=:top)
    hidespines!(ax22)
    hideydecorations!(ax22)

    ax3 = GLMakie.Axis(fig[2, 1], ylabel=L"\Delta e/e_0", xlabel=stepstring, yscale=log10)
    
    ax4 = GLMakie.Axis(fig[2, 2], ylabel=L"\Delta l/l_0", xlabel=stepstring, yscale=log10)
    
    ax5 = GLMakie.Axis(fig[2, 3], ylabel=L"g_{\mu \nu} u^\mu u^\nu", xlabel=stepstring)

    r = sol[1, 2]
    Δr = @views @inbounds abs.(sol[2:end, 2] .- r)
    mask = Δr .== 0.0
    Δr[mask] .= 1e-16
    nsteps = 1:N

    Δe = map(eachrow(sol)) do x
             energy(metric, x)
         end
    Δl = map(eachrow(sol)) do x
        ang_momentum(metric, x)
    end
    Δn = map(eachrow(sol)) do x
        lagrange(metric, x)
    end
    Δe = abs.(diff(Δe) ./ Δe[1])
    mask = Δe .== 0.0
    Δe[mask] .= 1e-16
    Δl = abs.(diff(Δl) ./ Δl[1])
    mask = Δl .== 0.0
    Δl[mask] .= 1e-16

    lines!(ax1, pl[1:2:end, 1], pl[1:2:end, 2], pl[1:2:end, 3])
    scatterlines!(ax6, λ, sol[:, 2])
    lines!(ax2, nsteps[1:end-1] ./ (10^pow), Δr)
    p = lines!(ax22, λ[1:end-1], Δr, alpha=0.001)
    delete!(ax22, p)
    
    lines!(ax3, nsteps[1:end-1] ./ (10^pow), Δe)
    lines!(ax4, nsteps[1:end-1] ./ (10^pow), Δl)
    lines!(ax5, nsteps ./ (10^pow), Δn)
    
    display(fig)
    save("errors_cap.png", fig)
    end
    println("Το robustness αποθηκεύτηκε επιτυχώς ως errors.png")
end

function c()
    metric = kerr_metric_bl
    pos = @SVector [0.0, 50.0, deg2rad(85.0), 1.0]
    g = metric(pos)
    gu = inv(g)

    rd = gu * [0.0, -1.0, 0.0, -3.0]
    rd = SVector{D,T}(vcat(0, rd[[2,3,4]]))
    ray_direction = get_final_velocity(rd, g, gu)
    println(ray_direction' * g * ray_direction)
    ray = MVector{2D,T}(vcat(pos, ray_direction))
    tspan = (0.0, 1000.0)
    # tol = 1e-12
    p = mnt(metric = metric, order = 7, flag = 0.0, save = true, E = energy(metric, ray), L = ang_momentum(metric, ray), is_thick_disk=false)
    prob = ODEProblem(geodesic!, ray, tspan, p)
    sims = solve(prob, Vern7(), callback=cb2, abstol=1e-15, reltol=1e-15, dtmax=0.001, maxiters=100000)    
    sol = reduce(hcat, sims.u)'
    λ = sims.t
    pl = reduce(hcat, map(eachrow(sol)) do x
                    @views @inbounds polar2cart(x[cart_mask])
                end)'
    N = size(sol, 1)
    pow = round(Int, log10(N))

    with_theme(Makie.theme_latexfonts(), fontsize=36) do 
    fig = Figure(size=(1920, 1080), pt_per_unit=1, px_per_unit=1)
    
    ax1 = Axis3(fig[1, 3])
    mesh!(ax1, Sphere(Point3f(0), 1.15f0), color = :black)
    xlims!(ax1, -5, 5)
    ylims!(ax1, -5, 5)
    zlims!(ax1, -5, 5)

    stepstring = latexstring(L"{\mathrm{steps}} \cdot 10^{"[1:end-1] * string(pow) * "}\$")
    
    ax6 = GLMakie.Axis(fig[1, 2], xlabel="λ", ylabel="r")

    ax2 = GLMakie.Axis(fig[1, 1], ylabel=L"\Delta r/r_0", xlabel=stepstring, yscale=log10)
    ax22 = GLMakie.Axis(fig[1, 1], xlabel="λ", yscale=log10, xaxisposition=:top)
    hidespines!(ax22)
    hideydecorations!(ax22)

    ax3 = GLMakie.Axis(fig[2, 1], ylabel=L"\Delta e/e_0", xlabel=stepstring, yscale=log10)
    
    ax4 = GLMakie.Axis(fig[2, 2], ylabel=L"\Delta l/l_0", xlabel=stepstring, yscale=log10)
    
    ax5 = GLMakie.Axis(fig[2, 3], ylabel=L"g_{\mu \nu} u^\mu u^\nu", xlabel=stepstring)

    r = sol[1, 2]
    Δr = @views @inbounds abs.(sol[2:end, 2] .- r)
    mask = Δr .== 0.0
    Δr[mask] .= 1e-16
    nsteps = 1:N

    Δe = map(eachrow(sol)) do x
             energy(metric, x)
         end
    Δl = map(eachrow(sol)) do x
        ang_momentum(metric, x)
    end
    Δn = map(eachrow(sol)) do x
        lagrange(metric, x)
    end
    Δe = abs.(diff(Δe) ./ Δe[1])
    mask = Δe .== 0.0
    Δe[mask] .= 1e-16
    Δl = abs.(diff(Δl) ./ Δl[1])
    mask = Δl .== 0.0
    Δl[mask] .= 1e-16

    lines!(ax1, pl[1:2:end, 1], pl[1:2:end, 2], pl[1:2:end, 3])
    scatterlines!(ax6, λ, sol[:, 2])
    lines!(ax2, nsteps[1:end-1] ./ (10^pow), Δr)
    p = lines!(ax22, λ[1:end-1], Δr, alpha=0.001)
    delete!(ax22, p)
    
    lines!(ax3, nsteps[1:end-1] ./ (10^pow), Δe)
    lines!(ax4, nsteps[1:end-1] ./ (10^pow), Δl)
    lines!(ax5, nsteps ./ (10^pow), Δn)
    
    display(fig)
    save("errors_V.png", fig)
    end
    println("Το robustness αποθηκεύτηκε επιτυχώς ως errors.png")
end

end