module PureDAQPMathOptInterfaceExt

"""
    PureDAQPMathOptInterfaceExt

`PureDAQP.Optimizer`, the MathOptInterface wrapper around [`ActiveSet`](@ref). The wrapper
itself is PureQPBase's and serves any algorithm; this supplies the one it runs and the name a
caller reaches it by.
"""

import MathOptInterface as MOI
import PureDAQP
import PureQPBase

# PureQPBase's own extension is not a dependency of this one, so it is not loaded while this
# is precompiled and the module is looked up when an optimizer is built instead. Both
# extensions are loaded by then: each needs only MathOptInterface, which the caller has.
function PureDAQP.Optimizer(; kwargs...)
    wrapper = Base.get_extension(PureQPBase, :PureQPBaseMathOptInterfaceExt)
    return wrapper.Optimizer{Float64}(PureDAQP.ActiveSet; kwargs...)
end

end
