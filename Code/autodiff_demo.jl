using ForwardDiff, GLMakie, LinearAlgebra, ProgressMeter


f(x) = [sin(x[1]) * x[2] + x[1]^3, exp(x[1]) + x[2]^2]
J(x) = [cos(x[1])*x[2] + 3*x[1]^2  sin(x[1]); exp(x[1])  2*x[2]]
x0 = [1.2, 0.7]

J_ad = ForwardDiff.jacobian(f, x0)
ad_error = norm(J(x0) - J_ad) 

function finite_diff_jacobian(f, x, h)
    n = length(x)
    m = length(f(x))
    J = zeros(m, n)
    for i in 1:n
        x_plus = copy(x); x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        J[:, i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return J
end

hs = 10.0 .^ range(-1, -12, length=50)
errors_fd = Float64[]

println("Υπολογισμός σφαλμάτων πεπερασμένων διαφορών...")
@showprogress for h in hs
    J_fd = finite_diff_jacobian(f, x0, h)
    push!(errors_fd, norm(J_fd - J(x0)))
end

errors_ad = fill(ad_error + 1e-16, length(hs))

fig = Figure(size=(1920, 1080))
set_theme!(theme_latexfonts(), fontsize = 20)
ax = Axis(fig[1,1], title="Automatic Differentiation vs Finite Differences", 
          xlabel="Stepsize h", ylabel="Absolute error", yscale=log10, xscale=log10)

lines!(ax, hs, errors_fd, color=:red, linewidth=3, label="Finite Diff.")
lines!(ax, hs, errors_ad, color=:blue, linewidth=3, linestyle=:dash, label="Auto. Diff.")
ylims!(5e-19,5e-4)
hlines!(ax, [1e-16], color=:gray, linestyle=:dot)
text!(ax, hs[end], 1e-16;
    text = L"10^{-16}",
    align = (:right, :bottom),    
    color = :gray,
    fontsize = 20,
    offset = (0, 3)              
)
axislegend(ax, position=:rb, L"\mathrm{expr=} (sin(x)y + x^3, e^x + y^2)^T")
save("FDvsAD.png", fig)

display(fig)