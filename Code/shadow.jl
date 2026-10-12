function brightness(pixel)
    return 0.299 * red(pixel) + 0.587 * green(pixel) + 0.114 * blue(pixel)
end

function map_brightness_to_alpha(pixel, factor)
    pixel = RGB(clamp(pixel.r*factor, 0, 1), clamp(pixel.g*factor, 0, 1), clamp(pixel.b*factor, 0, 1))
    bright = brightness(pixel)
    return RGBA(red(pixel), green(pixel), blue(pixel), bright)
end

function trace_rays_shadow(metric, sc::Int64, with_background::Bool=false)
    dist = T(1)

    N = sc
    M = sc 
    println("Grid Resolution: ", M, " x ", N)

    θ0 = deg2rad(87.5)
    alims = (-15, 15) 
    blims = (-15, 15)

    pos = SVector{D,T}([0, init_r, θ0, π/2])
    inputs = generate_rays(pos, metric, N, M, alims, blims, dist)

    prog[] = Progress(N * M)

    sols = tmap(inputs) do input
        sol, p_end = integrate_geodesics_scientific(input, metric, true)
        next!(prog[])
        return (sol, p_end)
    end

    println("Generating Shadow Rendering...")
    color_map = ColorSchemes.magma
    
    Is = map(sols) do x
        abs(x[2].Iv) 
    end
    
    mask = Is .== 0.0
    Is[mask] .= 1e-16
    Imax = maximum(Is)
    bl = minimum(Is)
    
    old_min = minimum(Is[.~mask])
    old_max = maximum(Is)
    
    colors = map(Is) do x
        if x == bl
            RGBA(0.0, 0.0, 0.0, 0.0)
        else
            new_min = 0.0
            new_max = 1.0
            new_range = new_max - new_min
            old_range = old_max - old_min
            y = (x - old_min) * new_range / old_range + new_min
            map_brightness_to_alpha(color_map[y], 2)
        end
    end

    img = map(sols) do sims
        pos_final = @views @inbounds sims[1]
        flag = sims[2].flag
        angs = @views @inbounds pos_final[2:3]
        
        if flag < 0.0
            return RGBA{Float64}(0.0, 0.0, 0.0, 1.0)
        else
            if with_background
                return RGBA(get_background_color(angs, background), 1.0)
            else
                return RGBA{Float64}(0.0, 0.0, 0.0, 1.0)
            end
        end
    end

    fin_img = img .+ colors
    fin_img .= map(fin_img) do pixel
        RGBA(clamp(pixel.r, 0, 1), clamp(pixel.g, 0, 1), clamp(pixel.b, 0, 1), clamp(pixel.alpha, 0, 1))
    end
    
    fin_img_2d = reshape(fin_img, M, N)

    set_theme!(theme_latexfonts())
    x_size = M + 150 < 800 ? 800 : M + 150
    y_size = N + 150 < 800 ? 800 : N + 150

    fig = Figure(size=(x_size, y_size), fontsize=26, pt_per_unit=1, px_per_unit=1)
    ax = GLMakie.Axis(fig[1,1], xlabel=L"\beta/M_{\odot}", ylabel=L"\alpha/M_{\odot}", aspect=DataAspect())
    ax.xgridvisible = false
    ax.ygridvisible = false
    
    image!(ax, blims, alims, fin_img_2d)
    Colorbar(fig[1,2], limits=(0.0, Imax), colormap=color_map, label=L"I")
    
    display(fig)
    filename = with_background ? "thin_disk_$(N)x$(M).png" : "bh_shadow_$(N)x$(M).png"
    save(filename, fig)
    println("Η εικόνα αποθηκεύτηκε επιτυχώς ως: ", filename)
end