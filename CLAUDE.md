# PureQP.jl

## Terminology: unmaterialized is not matrix-free

The two terms name different things. Both occur in this project. Do not use one for the other, and
do not infer one from the other.

**Unmaterialized** (also "structured"): a matrix held as a structured operator and never formed as
a dense array. The representations are `LinearMaps` maps (`KroneckerMap`, `BlockMap`, `FillMap`,
`WrappedMap`, ...) and the base's own `KroneckerOperator`, `JoinedOperator`, `StackedOperator`,
`Fill`, `Diagonal`. The point is speed for matrices that are too dense for a sparse representation to
pay and that have structure a dense array wastes. The solvers assemble the reduced matrix from the
structure and factor it directly (`ProductReduced`, `KroneckerReduced`, block and low-rank
backends). It is a direct method.

**Matrix-free**: only products with an operator are used, never its entries, and the linear system
is solved iteratively with Krylov methods (`linsys = :indirect`, which needs Krylov.jl). It is an
iterative method, and the ladder's last rung.

An unmaterialized input can reach either kind of backend. Which one it reaches is decided by the
selection ladder, not by the word used for the input. A statement about "unmaterialized"
performance concerns the structured assembly and the direct factorization, not Krylov iteration
counts or preconditioners.
