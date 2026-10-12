function uv_mapping(φ, θ)
    u = 1.0 - mod((φ - π) / π, 1.0)
    v = 1.0 - (mod(θ, π) / π) 
    return Vec2f(u, v)
end

function plot_celestial_sphere(ax, radius, rot_matrix, background)
    ϕ_pol = LinRange(0, π, 100)       

    # ==========================================
    # 1. ΠΙΣΩ ΗΜΙΣΦΑΙΡΙΟ (Κανονικό, Αδιαφανές)
    # ==========================================
    θ_az_back = LinRange(π, 2π, 100) 
    x_b = [radius * sin(ϕp) * cos(θa) for ϕp in ϕ_pol, θa in θ_az_back]
    y_b = [radius * sin(ϕp) * sin(θa) for ϕp in ϕ_pol, θa in θ_az_back]
    z_b = [radius * cos(ϕp) for ϕp in ϕ_pol, θa in θ_az_back]

    points_b = vec([Point3f(xv, yv, zv) for (xv, yv, zv) in zip(x_b, y_b, z_b)])
    faces_b = decompose(QuadFace{GLIndex}, Tesselation(Rect(0, 0, 1, 1), size(z_b)))
    normals_b = normalize.(points_b)

    uv_b = map(zip(x_b, y_b, z_b)) do point
        xv, yv, zv = point
        r = norm(point)
        polar = acos(clamp(zv / r, -1.0, 1.0))
        azimuthal = atan(yv, xv)
        azimuthal = azimuthal < 0 ? azimuthal + 2π : azimuthal
        return uv_mapping(azimuthal, polar)
    end

    gb_mesh_back = GeometryBasics.Mesh(meta(points_b; uv=Buffer(vec(uv_b)), normals=normals_b), faces_b)
    mesh!(ax, gb_mesh_back, color = Sampler(background))


    # ==========================================
    # 2. ΜΠΡΟΣΤΑ ΗΜΙΣΦΑΙΡΙΟ (Διάφανος Θόλος)
    # ==========================================
    θ_az_front = LinRange(0, π, 100) 
    x_f = [radius * sin(ϕp) * cos(θa) for ϕp in ϕ_pol, θa in θ_az_front]
    y_f = [radius * sin(ϕp) * sin(θa) for ϕp in ϕ_pol, θa in θ_az_front]
    z_f = [radius * cos(ϕp) for ϕp in ϕ_pol, θa in θ_az_front]

    points_f = vec([Point3f(xv, yv, zv) for (xv, yv, zv) in zip(x_f, y_f, z_f)])
    faces_f = decompose(QuadFace{GLIndex}, Tesselation(Rect(0, 0, 1, 1), size(z_f)))
    normals_f = normalize.(points_f)

    uv_f = map(zip(x_f, y_f, z_f)) do point
        xv, yv, zv = point
        r = norm(point)
        polar = acos(clamp(zv / r, -1.0, 1.0))
        azimuthal = atan(yv, xv)
        azimuthal = azimuthal < 0 ? azimuthal + 2π : azimuthal
        return uv_mapping(azimuthal, polar)
    end

    # Δημιουργούμε ένα διάφανο αντίγραφο της εικόνας (15% ορατότητα)
    bg_transparent = map(c -> RGBA{Float32}(c.r, c.g, c.b, 0.15f0), background)
    
    gb_mesh_front = GeometryBasics.Mesh(meta(points_f; uv=Buffer(vec(uv_f)), normals=normals_f), faces_f)
    mesh!(ax, gb_mesh_front, color = Sampler(bg_transparent), transparency=true)

    return ax
end

function plot_accretion_disk(ax=nothing)
    r = LinRange(3, 6, 100)
    θ = LinRange(0, 2pi, 100)
    x = r .* cos.(θ')
    y = r .* sin.(θ')
    z = zeros(size(y))
    
    if isnothing(ax)
        fig = Figure()
        ax = Axis3(fig[1,1])
        surface!(ax, x, y, z, color=fill(:white, size(z)...))
        return fig
    else
        surface!(ax, x, y, z, color=fill(:white, size(z)...))
    end
end