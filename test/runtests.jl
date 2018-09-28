# This file is part of the TaylorIntegration.jl package; MIT licensed

testfiles = (
    "one_ode.jl",
    "many_ode.jl",
    "complex.jl",
    "jettransport.jl",
    "lyapunov.jl",
    "bigfloats.jl",
    "common.jl",
    "rootfinding.jl",
    "taylorize.jl",
    "jac_lyapunov_taylorize.jl"
    )

for file in testfiles
    include(file)
end
