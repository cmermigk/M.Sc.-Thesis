
using GLMakie
using OrdinaryDiffEq
using DiffEqCallbacks
using StaticArrays
using LinearAlgebra
using ForwardDiff

function draw_twisted_grid_2d!(ax, a, r_H; max_r=5.2)
    for r in range(r_H, max_r, length=12)
        xs = [r * cos(φ) for φ in range(0, 2π, length=100)]
        ys = [r * sin(φ) for φ in range(0, 2π, length=100)]
        lines!(ax, xs, ys, color=(:gray, 0.3), linewidth=1.5)
    end
    
    for φ in range(0, 2π, length=24)
        xs = Float64[]
        ys = Float64[]
        for r in range(r_H, max_r, length=50)
            twist = -2.5 * a * (r_H / r)^3 
            push!(xs, r * cos(φ + twist))
            push!(ys, r * sin(φ + twist))
        end
        lines!(ax, xs, ys, color=(:gray, 0.4), linewidth=1.5)
    end
end

function compare_orbits()
    metrics = [
        (a = 0.998, color = :cyan, name = "Kerr (a = 0.998) - Escape"),
        (a = 0.500, color = :orange, name = "Kerr (a = 0.500) - Fall"),
        (a = 0.000, color = :red, name = "Kerr (a = 0.0) - Fall")
    ]

    pos = @SVector [0.0, 5.0, deg2rad(85.0), 0.0]

    set_theme!(GLMakie.theme_latexfonts())

    for m in metrics
        r_H = 1.0 + sqrt(1.0 - m.a^2)
        
        b_param = -2.005
        
        fig = Figure(size=(1600, 800), fontsize=26)
        
        ax2d_left = Axis(fig[1, 1], title="$(m.name)", aspect=DataAspect())
        
        bh_x = [r_H * cos(t) for t in range(0, 2π, length=100)]
        bh_y = [r_H * sin(t) for t in range(0, 2π, length=100)]
        poly!(ax2d_left, Point2f.(bh_x, bh_y), color = :black)
        
        draw_twisted_grid_2d!(ax2d_left, m.a, r_H, max_r=5.2)
        
        xlims!(ax2d_left, -5.2, 5.2)
        ylims!(ax2d_left, -5.2, 5.2)

        ax2d_right = Axis(fig[1, 2], 
                    title="r(λ)",
                    xlabel=L"\lambda", 
                    ylabel=L"r")

        metric_func = x -> BHI.kerr_metric_bl(x, m.a)
        g = metric_func(pos)
        gu = inv(g)
        rd = gu * [0.0, -1.0, 0.0, b_param]
        rd = SVector{BHI.D, BHI.T}(vcat(0, rd[2:4]))
        
        ray_direction = BHI.get_final_velocity(rd, g, gu)
        ray = MVector{2*BHI.D, BHI.T}(vcat(pos, ray_direction))
        
        p = BHI.mnt(metric = metric_func, order = 7, flag = 0.0, save = true, 
                    E = BHI.energy(metric_func, ray), L = BHI.ang_momentum(metric_func, ray), 
                    is_thick_disk = true, Iv = 0.0, u0 = MVector{BHI.D, BHI.T}(zeros(BHI.T,4)))
        
        prob = ODEProblem(BHI.geodesic!, ray, (0.0, 1500.0), p)
        
        function local_proj_affect!(integrator)
            u = integrator.u
            if any(!isfinite, u)
                integrator.p.flag = -1.0; terminate!(integrator); return
            end
            dt = integrator.dt
            p_int = integrator.p
            g_c = BHI.constraints 
            
            try
                g_prime = ForwardDiff.jacobian(x -> g_c(x, p_int), u)
                if any(!isfinite, g_prime)
                    integrator.p.flag = -1.0; terminate!(integrator); return
                end
                inv_g_sq = inv(g_prime * g_prime')
                n = size(g_c(u, p_int), 1)
                λ_proj = zeros(n); Δλ = zeros(n)
                j = 1; tol = 100 * dt^(p_int.order + 1)
                
                while true
                    if mod(j, 50) == 0
                        g_prime .= ForwardDiff.jacobian(x -> g_c(x, p_int), u)
                        inv_g_sq .= inv(g_prime * g_prime')
                    end
                    test_u = u + g_prime' * λ_proj
                    if any(!isfinite, test_u)
                        integrator.p.flag = -1.0; terminate!(integrator); return
                    end
                    g_val = g_c(test_u, p_int)
                    if any(!isfinite, g_val)
                        integrator.p.flag = -1.0; terminate!(integrator); return
                    end
                    Δλ = -inv_g_sq * g_val
                    λ_proj += Δλ
                    if any(!isfinite, Δλ)
                        integrator.p.flag = -1.0; terminate!(integrator); return
                    end
                    if norm(Δλ) < max(tol, 2e-16) || j > 500
                        break
                    end
                    j += 1
                end
                
                unew = u .+ g_prime' * λ_proj
                if any(!isfinite, unew)
                    integrator.p.flag = -1.0; terminate!(integrator); return
                end
                copyto!(integrator.u, unew)
                integrator.u_modified = true

                p1 = @views @inbounds BHI.polar2cart(integrator.uprev[BHI.cart_mask])
                p2 = @views @inbounds BHI.polar2cart(integrator.u[BHI.cart_mask])
                dist = norm(p1 - p2)
                
                if BHI.get_intersection(p1, p2, p_int.is_thick_disk)
                    integrator.p.flag += 1.0
                    if p_int.is_thick_disk
                        integrator.p.Iv += BHI.intensity_at_point(integrator.u, p_int) * dt
                    else
                        terminate!(integrator) 
                    end
                elseif @views @inbounds unew[2] < r_H + 0.005 || (dist > 5 && integrator.uprev[2] < 2)
                    integrator.p.flag = -1.0; terminate!(integrator) 
                elseif @views @inbounds unew[2] > 5.1
                    terminate!(integrator) 
                end
            catch
                integrator.p.flag = -1.0; terminate!(integrator); return
            end
        end
        
        local_cb2 = DiscreteCallback((u, λ, integrator) -> true, local_proj_affect!)
        
        sims = solve(prob, Vern7(), callback=local_cb2, abstol=1e-13, reltol=1e-13, dtmax=0.05, maxiters=500000)
        
        sol = reduce(hcat, sims.u)'
        λ_vals = sims.t
        r_vals = sol[:, 2]
        
        pl = reduce(hcat, map(eachrow(sol)) do x
            @views @inbounds BHI.polar2cart(x[BHI.cart_mask])
        end)'
        
        distances = [norm(pl[k, :]) for k in 1:size(pl, 1)]
        mask = distances .< 5.5
        
        x_full = pl[mask, 1]
        y_full = pl[mask, 2]
        λ_full = λ_vals[mask]
        r_full = r_vals[mask]
        
        obs_xy = Observable([Point2f(x_full[1], y_full[1])])
        obs_λr = Observable([Point2f(λ_full[1], r_full[1])])

        lines!(ax2d_left, obs_xy, color=m.color, linewidth=5.0, label="a = $(m.a)")
        axislegend(ax2d_left, position=:rt, labelsize=24)

        lines!(ax2d_right, obs_λr, color=m.color, linewidth=4.0, label="Trajectory")
        hlines!(ax2d_right, [r_H], color=:black, linestyle=:dash, linewidth=2.0, label=L"r_H \approx %$(round(r_H, digits=2))")
        
        xlims!(ax2d_right, 0.0, maximum(λ_full))
        ylims!(ax2d_right, 0.0, 5.2)
        axislegend(ax2d_right, position=:rt, labelsize=20)
        
        output_file = "frame_dragging_animation_a$(m.a).mp4"
        println("Ξεκινά η εγγραφή του βίντεο: $output_file ...")
        
        frames_step = max(1, length(x_full) ÷ 150)
        
        video_stream = VideoStream(fig, framerate = 30)
        
        for k in 2:frames_step:length(x_full)
            obs_xy[] = Point2f.(x_full[1:k], y_full[1:k])
            obs_λr[] = Point2f.(λ_full[1:k], r_full[1:k])
            recordframe!(video_stream)
        end
        
        save(output_file, video_stream)
        println("Το βίντεο αποθηκεύτηκε επιτυχώς ως $output_file")
    end
    println("Όλες οι προσομοιώσεις ολοκληρώθηκαν και αποθηκεύτηκαν!")
end