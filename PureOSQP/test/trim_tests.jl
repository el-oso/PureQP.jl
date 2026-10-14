@testitem "the public entry points are --trim compatible" begin
    using TrimCheck
    # `juliac --trim` needs every call resolved statically. This is what stops the solver
    # from silently acquiring a dynamic dispatch or a reflective call.
    #
    # `not_trimmable` is a negative control: it calls `Base.return_types`, which the
    # trimmer cannot resolve. If the checker passed that one it would not be
    # discriminating, and its verdict on the real entry points would mean nothing.
    entry = joinpath(@__DIR__, "trim", "entrypoints.jl")
    names = [
        :solve_default, :solve_polish, :solve_kkt, :solve_multiplier, :solve_unscaled, :solve_verbose,
        :solve_indirect, :solve_indirect_preconditioned, :solve_ipm_indirect, :solve_accelerated,
        :settings_and_rho, :derivatives,
        :solve_interruptible,
        :solve_time_limited, :solve_profiled,
        :setup_solve_update, :warm_started, :not_trimmable,
        :solve_ipm_default, :solve_ipm_kkt, :solve_ipm_unscaled, :solve_ipm_polish,
        :solve_ipm_verbose, :solve_ipm_verbose_indirect,
        :setup_ipm_update, :derivatives_ipm,
    ]
    sigs = [
        :(TrimEntry.$f(Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}))
            for f in names
    ]
    # The diagonal backend is selected by the matrix type, so its entry point takes
    # `Diagonal` and cannot share the signature the others are checked with.
    push!(names, :solve_diagonal)
    push!(
        sigs,
        :(
            TrimEntry.solve_diagonal(
                TrimEntry.DM, Vector{Float64}, TrimEntry.DM, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_ipm_diagonal)
    push!(
        sigs,
        :(
            TrimEntry.solve_ipm_diagonal(
                TrimEntry.DM, Vector{Float64}, TrimEntry.DM, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_tridiagonal)
    push!(
        sigs,
        :(
            TrimEntry.solve_tridiagonal(
                TrimEntry.STM, Vector{Float64}, TrimEntry.DM, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_tridiagonal_unsym)
    push!(
        sigs,
        :(
            TrimEntry.solve_tridiagonal_unsym(
                TrimEntry.TM, Vector{Float64}, TrimEntry.DM, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_banded)
    push!(
        sigs,
        :(
            TrimEntry.solve_banded(
                TrimEntry.STM, Vector{Float64}, TrimEntry.TM, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_lowrank)
    push!(
        sigs,
        :(
            TrimEntry.solve_lowrank(
                TrimEntry.DM, Vector{Float64}, TrimEntry.RC, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_kronecker)
    push!(
        sigs,
        :(
            TrimEntry.solve_kronecker(
                TrimEntry.DM, Vector{Float64}, TrimEntry.KO, Vector{Float64}, Vector{Float64}
            )
        )
    )
    # A keyword on a pair whose ladder backend is a fourth type is what merges past the union
    # length cap, so a second such pair keeps the guard off any one backend.
    push!(names, :solve_lowrank_scaled)
    push!(
        sigs,
        :(
            TrimEntry.solve_lowrank_scaled(
                TrimEntry.DM, Vector{Float64}, TrimEntry.RC, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_stacked)
    push!(
        sigs,
        :(
            TrimEntry.solve_stacked(
                Matrix{Float64}, Vector{Float64}, TrimEntry.SK, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_ipm_kronecker)
    push!(
        sigs,
        :(
            TrimEntry.solve_ipm_kronecker(
                TrimEntry.DM, Vector{Float64}, TrimEntry.KO, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :setup_kronecker)
    push!(
        sigs,
        :(
            TrimEntry.setup_kronecker(
                TrimEntry.DM, Vector{Float64}, TrimEntry.KO, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_block)
    push!(
        sigs,
        :(
            TrimEntry.solve_block(
                TrimEntry.BD, Vector{Float64}, TrimEntry.BD, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_operator)
    push!(
        sigs,
        :(
            TrimEntry.solve_operator(
                TrimEntry.PO, Vector{Float64}, TrimEntry.PO, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_ipm_operator)
    push!(
        sigs,
        :(
            TrimEntry.solve_ipm_operator(
                TrimEntry.PO, Vector{Float64}, TrimEntry.PO, Vector{Float64}, Vector{Float64}
            )
        )
    )
    push!(names, :solve_sciml)
    push!(
        sigs,
        :(
            TrimEntry.solve_sciml(
                TrimEntry.SO, Vector{Float64}, TrimEntry.SO, Vector{Float64}, Vector{Float64}
            )
        )
    )
    # Sparse operands, which are compatible only with the backend named -- see the comment in
    # `trim/entrypoints.jl` for why `:auto` on a doubly sparse pair is not.
    for f in (:solve_sparse_kkt, :solve_sparse_dense, :solve_ipm_sparse_kkt)
        push!(names, f)
        push!(
            sigs,
            :(
                TrimEntry.$f(
                    TrimEntry.SPM, Vector{Float64}, TrimEntry.SPM, Vector{Float64}, Vector{Float64}
                )
            )
        )
    end
    push!(names, :solve_sparse_diagonal)
    push!(
        sigs,
        :(
            TrimEntry.solve_sparse_diagonal(
                TrimEntry.DM, Vector{Float64}, TrimEntry.SPM, Vector{Float64}, Vector{Float64}
            )
        )
    )
    results = TrimCheck.validate(
        sigs...; init = :(include($entry); using .TrimEntry), progressbar = false
    )
    ok = Dict(
        String(f) => occursin("is trim compatible", sprint(show, r))
            for (f, r) in zip(names, results)
    )
    # A bare failed assertion names the entry point but not what the trimmer objected to,
    # which is the only part that says where to look.
    for (f, r) in zip(names, results)
        f === :not_trimmable || ok[String(f)] || println("$f:\n", sprint(show, r))
    end
    for f in names
        f === :not_trimmable && continue
        @test ok[String(f)]
    end
    @test !ok["not_trimmable"]
end
