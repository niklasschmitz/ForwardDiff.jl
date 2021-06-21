module FFTTest

using Test
using ForwardDiff
using ForwardDiff: Dual, valtype, value, partials
using FFTW
using AbstractFFTs: complexfloat, realfloat


x1 = Dual.(1:4.0, 2:5, 3:6)

@test value.(x1) == 1:4
@test partials.(x1, 1) == 2:5

@test complexfloat(x1)[1] === complexfloat(x1[1]) === Dual(1.0, 2.0, 3.0) + 0im
@test realfloat(x1)[1] === realfloat(x1[1]) === Dual(1.0, 2.0, 3.0)

@test fft(x1, 1)[1] isa Complex{<:Dual}

@testset "$f" for f in [fft, ifft, rfft, bfft]
    @test value.(f(x1)) == f(value.(x1))
    @test partials.(f(x1), 1) == f(partials.(x1, 1))
end

@testset "Dual * Plan over Duals" begin
   s = ForwardDiff.Dual(2.0, 1.0)
   x = Dual.(1:5.0, 2:6.0)
   p = s * FFTW.plan_fft(x)
   @test typeof(p*x) == Vector{Complex{ForwardDiff.Dual{Nothing, Float64, 1}}}
   @test partials.(s * fft(x)) ≈ partials.(p * x)
   @test value.(s * fft(x)) ≈ value.(p * x)
end

end # module
