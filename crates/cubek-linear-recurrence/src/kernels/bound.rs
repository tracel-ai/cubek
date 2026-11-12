//! How a tensor binds under a launcher.

use cubecl::tile::{Axis, Launcher, TileTensorLaunch};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// `tensor` bound under `launcher` along `axes`, `width` values a line.
pub(crate) fn bound(
    launcher: &Launcher,
    tensor: &PlainTensor,
    axes: &[Axis],
    width: usize,
) -> TileTensorLaunch {
    launcher
        .arg(tensor.binding.clone(), tensor.elem)
        .axes(axes)
        .vectorize(width)
        .build()
        .expect(ADMITTED_PLAN_RUNS)
        .arg()
}
