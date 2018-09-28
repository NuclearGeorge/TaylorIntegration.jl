# The following tests are an attempt to @taylorize the evaluation of the
# user-provided Jacobian function for the computation of Lyapunov spectra

using Test

using TaylorIntegration
import LinearAlgebra: I, tr

@testset "Test `stabilitymatrix!`" begin
    q0 = rand(3) # [19.0, 20.0, 50.0]; #the initial condition
    t0 = 0.0 #the initial time
    dof = length(q0) # degrees of freedom
    Q0 = Matrix{Float64}(I, dof, dof) # dof x dof identity matrix
    x0 = vcat(q0, reshape(Q0, dof*dof)); # initial conditions: eqs of motion and variationals

    #Lorenz system parameters
    const σ = 16.0
    const β = 4.0
    const ρ = 45.92

    #Taylor1 variables for evaluation of eqs of motion
    const _order = 28
    t = t0+Taylor1(_order)
    x = Taylor1.(x0, _order);
    dx = similar(x);

    @taylorize function lorenz!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    TaylorIntegration.jetcoeffs!(t, view(x, 1:dof), view(dx, 1:dof), Val(lorenz!))

    function lorenz_jac!(t, x, jac)
        jac[1,1] = -σ+zero(x[1]); jac[1,2] = σ+zero(x[1]); jac[1,3] = zero(x[1])
        jac[2,1] = ρ-x[3]; jac[2,2] = -1.0+zero(x[1]); jac[2,3] = -x[1]
        jac[3,1] = x[2]; jac[3,2] = x[1]; jac[3,3] = -β+zero(x[1])
        nothing
    end

    # Compute Jacobian using previously defined `lorenz_jac!` function
    jac = Matrix{Taylor1{eltype(q0)}}(undef, dof, dof);
    lorenz_jac!(t, x, jac)
    @time lorenz_jac!(t, x, jac)

    xi = set_variables("δ", order=1, numvars=3);
    δx = Array{TaylorN{Taylor1{Float64}}}(undef, 3);
    dδx = similar(δx);
    q0T = Taylor1.(q0,_order);
    _δv = Array{TaylorN{Taylor1{Float64}}}(undef, 3);
    for ind in 1:3
        _δv[ind] = one(t)*TaylorN(Taylor1{Float64}, ind, order=1)
    end
    jac2 = similar(jac);
    TaylorIntegration.stabilitymatrix!(lorenz!, t, x[1:dof], δx, dδx, jac2, _δv)
    @test jac2 == jac

    # ex = :(function lorenz_jac_parsed!(t, x, jac)
    #     jac[1,1] = -σ+zero(x[1]); jac[1,2] = σ+zero(x[1]); jac[1,3] = zero(x[1])
    #     jac[2,1] = ρ-x[3]; jac[2,2] = -1.0+zero(x[1]); jac[2,3] = -x[1]
    #     jac[3,1] = x[2]; jac[3,2] = x[1]; jac[3,3] = -β+zero(x[1])
    #     nothing
    # end)
    #
    # @eval $ex
    #
    # TaylorIntegration._make_parsed_jetcoeffs(ex)

    # output from _make_parsed_jetcoeffs, modify:
    # - jac::AbstractVector -> jac::AbstractMatrix
    # - comment recursion relations
    # - fix order loop: for ord = 1:order-1 -> for ord = 1:order
    function TaylorIntegration.jetcoeffs!(t::Taylor1{T}, x::AbstractVector{Taylor1{S}}, jac::AbstractMatrix{Taylor1{S}}, ::Val{lorenz_jac_parsed!}) where {T <: Real, S <: Number}
          order = t.order
          tmp446 = Taylor1(-(constant_term(σ)), order)
          tmp447 = Taylor1(zero(constant_term(x[1])), order)
          jac[1, 1] = Taylor1(constant_term(tmp446) + constant_term(tmp447), order)
          tmp449 = Taylor1(zero(constant_term(x[1])), order)
          jac[1, 2] = Taylor1(constant_term(σ) + constant_term(tmp449), order)
          jac[1, 3] = Taylor1(zero(constant_term(x[1])), order)
          jac[2, 1] = Taylor1(constant_term(ρ) - constant_term(x[3]), order)
          tmp454 = Taylor1(zero(constant_term(x[1])), order)
          jac[2, 2] = Taylor1(constant_term(-1.0) + constant_term(tmp454), order)
          jac[2, 3] = Taylor1(-(constant_term(x[1])), order)
          jac[3, 1] = Taylor1(identity(constant_term(x[2])), order)
          jac[3, 2] = Taylor1(identity(constant_term(x[1])), order)
          tmp457 = Taylor1(-(constant_term(β)), order)
          tmp458 = Taylor1(zero(constant_term(x[1])), order)
          jac[3, 3] = Taylor1(constant_term(tmp457) + constant_term(tmp458), order)
          # for __idx = eachindex(x)
          #     (x[__idx]).coeffs[2] = (jac[__idx]).coeffs[1]
          # end
          for ord = 1:order
              ordnext = ord + 1
              TaylorSeries.subst!(tmp446, σ, ord)
              TaylorSeries.zero!(tmp447, x[1], ord)
              TaylorSeries.add!(jac[1, 1], tmp446, tmp447, ord)
              TaylorSeries.zero!(tmp449, x[1], ord)
              TaylorSeries.add!(jac[1, 2], σ, tmp449, ord)
              TaylorSeries.zero!(jac[1, 3], x[1], ord)
              TaylorSeries.subst!(jac[2, 1], ρ, x[3], ord)
              TaylorSeries.zero!(tmp454, x[1], ord)
              TaylorSeries.add!(jac[2, 2], -1.0, tmp454, ord)
              TaylorSeries.subst!(jac[2, 3], x[1], ord)
              TaylorSeries.identity!(jac[3, 1], x[2], ord)
              TaylorSeries.identity!(jac[3, 2], x[1], ord)
              TaylorSeries.subst!(tmp457, β, ord)
              TaylorSeries.zero!(tmp458, x[1], ord)
              TaylorSeries.add!(jac[3, 3], tmp457, tmp458, ord)
              # for __idx = eachindex(x)
              #     (x[__idx]).coeffs[ordnext + 1] = (jac[__idx]).coeffs[ordnext] / ordnext
              # end
          end
          return nothing
      end

    jac3 = similar(jac)

    TaylorIntegration.jetcoeffs!(t, view(x, 1:dof), jac3, Val(lorenz_jac_parsed!))
    @time TaylorIntegration.jetcoeffs!(t, view(x, 1:dof), jac3, Val(lorenz_jac_parsed!))

    @test jac3 == jac
end
