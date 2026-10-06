using TestItemRunner

# Items tagged `:gpu` are not run. They exercise the GPU array path through JLArrays, and
# compiling that path crashes Julia 1.13.0 inside LLVM's loop vectorizer on AVX-512 targets.
#
# Items tagged `:moi` run the MathOptInterface conformance suite, which is about half of this
# suite's running time. `PUREQP_SKIP_MOI=true` leaves them out, so a build can check the
# wrapper against one Julia version rather than every one of them.
const SKIP_MOI = get(ENV, "PUREQP_SKIP_MOI", "false") == "true"

@run_package_tests filter = ti -> !(:gpu in ti.tags) && !(SKIP_MOI && :moi in ti.tags)
