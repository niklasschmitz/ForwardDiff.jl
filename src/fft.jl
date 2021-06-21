
ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) =
    Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N

ForwardDiff.tagtype(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = T
ForwardDiff.tagtype(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = T

# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region; kwargs...)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region; kwargs...)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region; kwargs...)

    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:ForwardDiff.Dual}) = 
            (Y .= _apply_plan(p, X))
        
        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            (Y .= _apply_plan(p, X))
    end
end

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    T = ForwardDiff.tagtype(eltype(x))
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual{T}(real(val), map(real, parts)),
            ForwardDiff.Dual{T}(imag(val), map(imag, parts)),
        )
    end
end

function _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray) where {T,P}
    _apply_plan(p.p, p.scale * x) # for when p.scale is Dual, need out-of-place
end
