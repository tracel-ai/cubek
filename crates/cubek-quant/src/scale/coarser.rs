//! The coarser level of a two-level scheme, which every finest scale under it is normalized by.

use cubecl::prelude::*;

use crate::layout::{
    ScalesLayout, ScalesLayoutArgs, ScalesView, ScalesViewLaunch, ScalesViewMut, scales_layout,
    scales_view,
};
use crate::scale::Scale;
use crate::scheme::QuantScheme;

/// The finest level's scale grid and, for a two-level scheme, the coarser level's.
/// Bindings arrive finest first, the order `check_scale_bindings` counts.
pub(crate) fn split_levels(scales: &[TensorBinding]) -> (TensorBinding, Option<TensorBinding>) {
    (scales[0].clone(), scales.get(1).cloned())
}

/// The coarser scale of one value's block, absent for the schemes that have none.
///
/// Bound as f32 whatever the finest level stores: cubecl's `check_scale_bindings` refuses anything
/// narrower.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CoarserScale {
    value: ComptimeOption<f32>,
}

impl CoarserScale {
    /// The launch view of `scheme`'s coarser grid, bound as `grid`, over the quantized `values`;
    /// none for a scheme with one level.
    ///
    /// # Panics
    ///
    /// As [`scales_layout`](crate::layout::scales_layout).
    pub fn view(
        values: TensorBinding,
        grid: Option<TensorBinding>,
        scheme: &QuantScheme,
    ) -> Option<ScalesViewLaunch> {
        let level = scheme.levels().nth(1)?;
        Some(scales_view(values, grid?, 1, scheme, level.size))
    }

    /// The layout of `scheme`'s coarser grid, bound as `grid`, over the quantized `values`, which
    /// elects one writer a coarser block; none for a scheme with one level.
    ///
    /// # Panics
    ///
    /// As [`scales_layout`].
    pub fn layout(
        values: &TensorBinding,
        grid: Option<&TensorBinding>,
        scheme: &QuantScheme,
    ) -> Option<ScalesLayoutArgs> {
        let level = scheme.levels().nth(1)?;
        Some(scales_layout(values, grid?, 1, scheme, level.size))
    }
}

#[cube]
impl CoarserScale {
    /// The coarser scale of the value at `pos`, read out of the coarser level's grid.
    pub fn read(grid: &ComptimeOption<ScalesView<'_, f32>>, pos: usize) -> CoarserScale {
        let value = #[comptime]
        match grid {
            ComptimeOption::Some(grid) => ComptimeOption::new_Some(grid.read(pos)),
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        CoarserScale { value }
    }

    /// Copy the scale into the quantized tensor's own coarser grid, where dequantize reads it back
    /// from, once for its block: from the value at the start of it in `layout`, the coarser
    /// level's layout.
    ///
    /// The caller cannot hand its input buffer through instead: a quantized tensor's scales live
    /// in one allocation the tensor owns, so the scale has to land inside that allocation, and
    /// only the kernel writing it gets there without a second copy.
    pub fn write(
        &self,
        out: ComptimeOption<ScalesViewMut<'_, f32>>,
        layout: ComptimeOption<ScalesLayout>,
        pos: usize,
    ) {
        #[comptime]
        match out {
            // Both bindings are checked against the same scheme at launch, so somewhere to write
            // the scale means there is one to read, and a layout to elect its writer.
            ComptimeOption::Some(mut out) => {
                if layout.unwrap().is_block_start(pos) {
                    out.write(pos, self.value.unwrap());
                }
            }
            ComptimeOption::None => {}
        }
    }

    /// The factor a value under `finest` is scaled by. Below this pairing nothing knows how many
    /// levels the scheme spreads its scales across.
    pub fn at<FS: CubePrimitive>(&self, finest: FS) -> Scale<FS> {
        Scale::<FS> {
            inner: finest,
            coarser: self.value,
        }
    }
}
