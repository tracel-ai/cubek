use super::base::PlaneRmsNormMicrokernel;
use crate::rms_norm::{HeldRow, RmsNormScale, RowReader};
use crate::row_store::OwnerLines;
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Monoid, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's row: `out = input · scale · weight`, each lane of the plane its lines. `input`,
/// `gate` where the norm is gated, and `weight` are windowed to the row, and `out` is where the
/// plane writes it.
#[cube]
pub fn plane_rms_norm_microkernel<E: Float, H: Float, W: Float, N: Size>(
    input: &Tile,
    gate: &ComptimeOption<Tile>,
    weight: &Tile,
    out: &mut OwnerLines,
    scale: &RmsNormScale,
    #[comptime] microkernel: PlaneRmsNormMicrokernel,
) {
    let lane = UNIT_POS_PLANE as usize;
    let reader = RowReader::new(input, gate, microkernel.gate);
    let mut row = HeldRow::<H, N>::new(microkernel.held_row());
    let sum = UnitShare::Plane.reduce::<f32>(row.load::<E>(&reader, lane), Monoid::Sum);
    row.write::<E, W>(&reader, weight, out, lane, scale.of(sum));
}
