using OrdinaryDiffEq, LinearAlgebra, GLMakie, ForwardDiff
using ProgressMeter

# Εξίσωση κίνησης σε κύκλο: x' = -y, y' = x
function circle!(du, u, p, t)
    du[1] = -u[2]
    du[2] = u[1]
end



function constraints(u, p)
    return [u[1]^2 + u[2]^2 - 1.0]
end

condition(u, t, integrator) = true

function affect!(integrator)
    u = integrator.u
    p = integrator.p
    g = constraints

    g_prime = ForwardDiff.jacobian(x -> g(x, p), u)
    inv_g_sq = inv(g_prime * g_prime')

    n = size(g(u, p), 1)
    λ = zeros(n)
    Δλ = zeros(n)
    i = 1
    tol = 1e-12

    while true
        Δλ = -inv_g_sq * g(u + g_prime' * λ, p)
        λ += Δλ
        val = norm(Δλ)

        if val < tol || i > 100
            break
        end
        i += 1
    end

    unew = u .+ g_prime' * λ
    copyto!(integrator.u, unew)
    integrator.u_modified = true
end

cb = DiscreteCallback(condition, affect!)

dt = 0.3

u0 = [1.0, 0.0]
tspan = (0.0, 10000.0)
prob1 = ODEProblem(circle!, u0, tspan)
sol1 = solve(prob1, RK4(), dt=dt, adaptive=false, maxiters=1000000)

p = (order=4,)
prob2 = ODEProblem(circle!, u0, tspan, p)
sol2 = solve(prob2, RK4(), dt=dt, adaptive=false, callback=cb, maxiters=1000000)

r1 = sqrt.(sol1[1,:].^2 + sol1[2,:].^2)
r2 = sqrt.(sol2[1,:].^2 + sol2[2,:].^2)

fig = Figure(size=(1980, 960),pt_per_unit=1, px_per_unit=1,figure_padding = 60) # Λίγο πιο ψηλό για να χωρέσει 3 σειρές
set_theme!(GLMakie.theme_latexfonts(), fontsize=36)

Label(fig[0, 1:5], "RK4 | dt = $(dt)", color=:black)

ax1 = GLMakie.Axis(fig[1, 1:2], title="Trajectory")
θ = range(0, 2π, length=100)
lines!(ax1, cos.(θ), sin.(θ), color=:gray, linewidth=1, linestyle=:dash)

traj_red_obs = Observable(Point2f[])
traj_blue_obs = Observable(Point2f[])

lines!(ax1, traj_red_obs, color=:red, linewidth=3, label="without Projection")
lines!(ax1, traj_blue_obs, color=:blue, linewidth=3, label="with Projection")

axislegend(ax1, position=(:center, :center))

xlims!(ax1, -1.1, 1.1)   
ylims!(ax1, -1.1, 1.1)

ax2 = GLMakie.Axis(fig[1, 3:5], title="Radial behavior", xlabel="t", ylabel="r(t)")
ax2.aspect = nothing  
r_blue_obs = Observable(Point2f[])
r_red_obs = Observable(Point2f[])
lines!(ax2, r_blue_obs, color=:blue, linewidth=3, label="with Projection")
lines!(ax2, r_red_obs, color=:red, linewidth=3, label="without Projection")
axislegend(ax2,position=:lb)
xlims!(ax2,0.0,10000.0)
ylims!(ax2,0.825,1.005)

frames = 1:200
prog = Progress(length(frames), desc="Δημιουργία GIF...")

record(fig, "manifold_projection_demo.mp4", frames; framerate=30) do i
    progress = i / 200

    idx_blue = max(1, min(round(Int, progress *  length(sol2.t)), length(sol2.t)))
    idx_red = max(1, min(round(Int, progress * length(sol1.t)), length(sol1.t)))

    traj_blue_pts = [Point2f(sol2[1,j], sol2[2,j]) for j in 1:idx_blue]
    traj_red_pts = [Point2f(sol1[1,j], sol1[2,j]) for j in 1:idx_red]
    traj_blue_obs[] = traj_blue_pts
    traj_red_obs[] = traj_red_pts

    r_blue_pts = [Point2f(sol2.t[j], r2[j]) for j in 1:idx_blue]
    r_red_pts = [Point2f(sol1.t[j], r1[j]) for j in 1:idx_red]
    r_blue_obs[] = r_blue_pts
    r_red_obs[] = r_red_pts
    # autolimits!(ax2)
    next!(prog)
end

save("manifold_projection_static.png", fig)
display(fig)

println("Το GIF αποθηκεύτηκε ως manifold_projection_demo.mp4")
println("Η στατική εικόνα αποθηκεύτηκε ως manifold_projection_static.png")