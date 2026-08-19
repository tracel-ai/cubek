//! The one factor a two-level scheme normalizes every inner scale against.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};

use crate::scale::Scale;

/// The innermost level's scale grid and, for a two-level scheme, the tensor's global scale.
/// Bindings arrive innermost first, the order `check_scale_bindings` counts.
pub(crate) fn split_levels<R: Runtime>(
    scales: &[TensorBinding<R>],
) -> (TensorBinding<R>, Option<TensorBinding<R>>) {
    (scales[0].clone(), scales.get(1).cloned())
}

/// The scale a whole tensor is normalized by, absent for the schemes that have none.
///
/// Bound as f32 whatever the inner level stores: cubecl's `check_scale_bindings` refuses anything
/// narrower, since one scale for a whole tensor saves nothing by narrowing and only reintroduces
/// rounding error.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct GlobalScale {
    value: ComptimeOption<f32>,
}

#[cube]
impl GlobalScale {
    /// Load it out of its binding.
    ///
    /// Once per kernel, so a unit scaling several regions hoists the load out of its loop.
    pub fn read(binding: ComptimeOption<LinearView<'_, f32>>) -> GlobalScale {
        let value = #[comptime]
        match binding {
            ComptimeOption::Some(binding) => ComptimeOption::new_Some(binding.read(0)),
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        GlobalScale { value }
    }

    /// Copy the scale into the quantized tensor's own scale region, where dequantize reads it back
    /// from.
    ///
    /// The caller cannot hand its input buffer through instead: a quantized tensor's scales live
    /// in one allocation the tensor owns, so the scale has to land inside that allocation, and
    /// only the kernel writing it gets there without a second copy.
    pub fn write(&self, out: ComptimeOption<LinearViewMut<'_, f32>>) {
        #[comptime]
        match out {
            // Both bindings are checked against the same scheme at launch, so somewhere to write
            // the scale means there is one to read.
            ComptimeOption::Some(mut out) => {
                if ABSOLUTE_POS == 0 {
                    out.write(0, self.value.unwrap());
                }
            }
            ComptimeOption::None => {}
        }
    }

    /// The factor a value under `inner` is scaled by. Below this pairing nothing knows how many
    /// levels the scheme spreads its scales across.
    pub fn at<FS: CubePrimitive>(&self, inner: FS) -> Scale<FS> {
        Scale::<FS> {
            inner,
            global: self.value,
        }
    }
}
