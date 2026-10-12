using LinearAlgebra

function normalization(normal::AbstractArray,#Vector{D,T},
                            g::Metric, gu::Metric) where Metric

    @assert normal[1] == zero(eltype(normal))

    @fastmath begin
#     normal = SVector{D,T}(normal./norm(normal))
    t = gu * SVector{D,T}(1, 0, 0, 0)
    t2 = t' * g * t
    n2 = normal' * g * normal
    normal = SVector{D,T}((t / sqrt(-t2) + normal / sqrt(n2)))
    end
    normal
end

###### preps initial data #####
function generate_rays(pos, metric, height, width, alims, blims, dist)
    rays = Array{MVector{2D,Float64}}(undef, width, height)
    g = metric(pos)
    gu = inv(g)

    as = LinRange(alims[1], alims[2], height)
    bs = LinRange(blims[1], blims[2], width)

    for (j, a) in enumerate(as)
        for (i, b) in enumerate(bs)
            pt = -1.0 
            
      
            a_safe = a + T(1e-5)
            b_safe = b + T(1e-5)

            ray_direction = gu*[0.0, -1.0, a_safe*pt, -b_safe*pt]
            ray_direction = @views @inbounds SVector{D,T}(vcat(0, ray_direction[2:4]))
            
            rays[i, j] = vcat(pos, normalization(ray_direction, g, gu))
        end
    end
    return rays
end