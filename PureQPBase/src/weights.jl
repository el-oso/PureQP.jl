"""
    SystemWeights{T,V}

The diagonal weights and the primal regularization a [`LinearSystem`](@ref) is built from:

    reduced   P̃ + σI + Ãᵀ diag(w) Ã
    KKT       [P̃ + σI   Ãᵀ  ;  Ã   −diag(w_inv)]

Invariants, maintained by the owner: `w[i] > 0`, `w_inv[i] == inv(w[i])` as the owner
computed it, `sigma > 0`, `length(w) == m`. The vectors are read in place, so a change to
their contents reaches the next [`refactor_weights!`](@ref) without a new object. `sigma` is
assigned in place too, and a change to it reaches the next [`factorize!`](@ref), since the
part of a factorization `refactor_weights!` keeps may depend on it.

ADMM holds `w = ρ`, `w_inv = ρ⁻¹` and `sigma = σ`.
"""
mutable struct SystemWeights{T <: Real, V <: AbstractVector{T}}
    # Mutable with every field `const`, not an immutable `struct`: an immutable one is stored
    # inline in the workspace, and Julia 1.12 loads the vectors out of an inline field without
    # the non-null/dereferenceable annotations a load from a heap object carries. Without them
    # LLVM cannot hoist the data-pointer load out of a bounds-checked loop, and the elementwise
    # kernels `admm_step!` inlines over `w` and `w_inv` stop vectorizing.
    const w::V
    const w_inv::V
    # Not `const`: the interior-point method changes it on a regularization bump, and a new
    # object for that would allocate inside the iteration.
    sigma::T
end
