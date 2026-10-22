use cubek_tile::Launcher;
use cubek_tile::launch::Bound;

/// The operands of one launch, bound under its launcher; `Out` is a buffer's [`Bound`] or a
/// sink's `Unbound`. A float operand is one [`Bound`]; the dense product binds its weight as one
/// tile however it is stored, a packed one's words under its scales
/// ([`PackedTensor::bind`](crate::kernels::tensor::PackedTensor::bind)); the NVFP4 product binds its words
/// beside its scales (`Nvfp4Bound`). Not `Debug`: cubek's `Launcher` is not.
pub struct MatmulBinding<Out, Lhs = Bound, Rhs = Bound> {
    pub launcher: Launcher,
    pub lhs: Lhs,
    pub rhs: Rhs,
    pub out: Out,
}

impl<Out, Lhs, Rhs> MatmulBinding<Out, Lhs, Rhs> {
    pub fn map_out<T>(self, f: impl FnOnce(Out) -> T) -> MatmulBinding<T, Lhs, Rhs> {
        MatmulBinding {
            launcher: self.launcher,
            lhs: self.lhs,
            rhs: self.rhs,
            out: f(self.out),
        }
    }
}
