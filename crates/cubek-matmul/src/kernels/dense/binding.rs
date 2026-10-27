use crate::kernels::tensor::{PackedTensor, PlainTensor};
use cubecl::tile::launch::Bound;
use cubecl::tile::{Launcher, TileArgLaunch};
use cubek_microkernel::matmul::stored::BlockScales;
use cubek_std::matmul_axes::{K, N};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// The operands of one launch, bound under its launcher; `Out` is a buffer's [`Bound`] or a sink's
/// `Unbound`. A float operand is one [`Bound`]; the dense product binds each input as one tile
/// however it is stored, a packed one's words under its scales
/// ([`PackedTensor::bind`](crate::kernels::tensor::PackedTensor::bind)); the NVFP4 product binds
/// its words beside its scales. Not `Debug`: the tile DSL's `Launcher` is not.
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

/// A packed matrix and the float one it decodes into, bound under one launcher. Not `Debug`:
/// the tile DSL's `Launcher` is not.
pub struct MatmulDequantizeBinding {
    pub launcher: Launcher,
    /// The packed matrix as one tile: its words under their scales.
    pub packed: TileArgLaunch,
    pub out: Bound,
}

impl MatmulDequantizeBinding {
    /// `packed`, its words `width` values wide, beside `into`, under `launcher`: a float `[k, n]`
    /// matrix in dense rows of the axis the packed words do not run along, so each read of the
    /// words decodes into the line it writes.
    pub fn new(
        packed: &PackedTensor,
        into: &PlainTensor,
        launcher: Launcher,
        width: usize,
    ) -> Self {
        let operand = packed.operand();
        let lined = operand.lined();
        let across = match lined == K {
            true => N,
            false => K,
        };
        let (out, out_axes) = packed.restated(
            into,
            (across, packed.extent(across)),
            (lined, packed.extent(lined)),
        );
        let out = launcher
            .arg(out.binding, out.elem)
            .axes(&out_axes)
            .vectorize(operand.read(lined))
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let packed = packed.bind(&launcher, width, BlockScales::Fields);
        Self {
            launcher,
            packed,
            out,
        }
    }
}
