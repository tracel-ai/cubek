use super::base::CubeGatedDeltaConvMicrokernel;
use super::convolved_head::ConvolvedHead;
use super::qk_scale::QkScale;
use super::tiles::GatedDeltaConvTiles;
use crate::gated_delta::KEY_HEAD;
use cubecl::tile::Region;
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's key head of one row, or of one position of a sequence: the head's channels convolved
/// and activated, the newest values written as the ring's next column over a window, then the
/// queries scaled by `1 / (‖q‖ · √key_features)` and the keys by `1 / ‖k‖`, written with the values
/// as `f32`.
#[cube]
pub fn cube_gated_delta_conv_microkernel<E: Float, N: Size>(
    cube: &Region,
    tiles: &mut GatedDeltaConvTiles,
    epsilon: f32,
    #[comptime] microkernel: CubeGatedDeltaConvMicrokernel,
) {
    let mut convolved = ConvolvedHead::<N>::new(microkernel);
    convolved.convolve::<E>(cube, tiles, cube.coord(KEY_HEAD));
    sync_cube();
    let key_features = comptime!(microkernel.heads.key_features);
    QkScale::of::<N>(&convolved, epsilon, key_features).write::<N>(
        cube,
        &convolved,
        &mut tiles.query,
        &mut tiles.key,
        key_features,
    );
}
