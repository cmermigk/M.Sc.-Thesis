using StaticArrays

function kerr_metric_bl(xx::AbstractArray, a::Real=T(0.998))
    t, r, θ, φ = xx
    a = T(a) # Μετατρέπουμε σε Float64 για σιγουριά

    M = T(1.0)
    @fastmath begin
        Σ = r^2 + a^2 * cos(θ)^2
        Δ = r^2 - 2 * M * r + a^2
        gtt = -(1 - 2 * M * r / Σ)
        grr = Σ / Δ
        gθθ = Σ
        gφφ = (r^2 + a^2 + (2 * M * a^2 * r * sin(θ)^2) / Σ) * sin(θ)^2
        gtφ = -(2 * M * a * r * sin(θ)^2) / Σ
    end
    
    return @SMatrix [gtt 0 0 gtφ;
                     0 grr 0 0;
                     0 0 gθθ 0;
                     gtφ 0 0 gφφ]
end

function kerr_schild(xx::SVector{D,T})::SMatrix{D,D,T} where {T}
    M = 1

    t,x,y,z = xx
    #@assert !any(isnan, (t, x, y, z))
    @fastmath begin
    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = @SMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
    a = T(0.998)                       # T(0.8)

    ρ = sqrt(x^2 + y^2 + z^2)
    if ρ^2 - a^2<0
        r = NaN
    else
        r = sqrt( (ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2) )
    end
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = SVector{D,T}(1,
                     (r*x + a*y) / (r^2 + a^2),
                     (r*y - a*x) / (r^2 + a^2),
                     z / r)

    g = @SMatrix T[η[a,b] + f * k[a] * k[b] for a in 1:D, b in 1:D]
    end
    g
end
