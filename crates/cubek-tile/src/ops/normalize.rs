//! Elementwise normalization and the normalization policy shared by procedural filters.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{ir::Scope, unexpanded};

use crate::*;

/// How division handles a denominator whose magnitude is too small to divide by. Both fields are
/// comptime kernel constants. The fallback is the reciprocal multiplier, so the default maps a
/// guarded result to zero.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DivGuard {
    pub epsilon: f32,
    pub fallback: f32,
}

fn validate_guard(guard: DivGuard) {
    assert!(
        guard.epsilon.is_finite() && guard.epsilon >= 0.0,
        "DivGuard: epsilon must be finite and non-negative"
    );
}

/// Which taps contribute to the sum of a normalized separable filter factor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TapMask {
    /// Sum only taps whose projected input sample is in bounds, removing edge darkening. The
    /// contraction must read the original source window in place; a shared-memory stage no longer
    /// records which staged zeros came from outside that window and is rejected at expansion.
    #[default]
    Masked,
    /// Sum the full mathematical support, preserving the fade of zero padding at an edge.
    Unmasked,
}

/// A guarded reciprocal that preserves the sign of a valid denominator. Substitution keeps the
/// discarded division finite even though `select` evaluates both arms; NaN fails the comparison
/// and takes the fallback.
#[cube]
pub fn guarded_recip<E: Float>(d: E, #[comptime] guard: DivGuard) -> E {
    comptime!(validate_guard(guard));
    guarded_recip_numeric::<E>(d, guard)
}

/// The contraction remains generic over numeric weights for unnormalized recipes. Only the
/// float-only public normalization surface can set the flag that reaches this helper.
#[cube]
pub(crate) fn guarded_recip_numeric<E: Numeric>(d: E, #[comptime] guard: DivGuard) -> E {
    let epsilon = E::cast_from(comptime!(guard.epsilon));
    let fallback = E::cast_from(comptime!(guard.fallback));
    let valid = d.abs() > epsilon;
    let safe = select(valid, d, E::from_int(1));
    select(valid, E::from_int(1) / safe, fallback)
}

impl<T: Float> Tile<T> {
    /// Normalize a separable procedural tile's factor runs where the gather contraction evaluates
    /// them. This is deliberately refused for opaque recipes and backed tiles: silently routing
    /// either through a post-pass would conceal an extra contraction or memory walk. A masked
    /// normalization also requires the contraction's rhs to remain at its original source window;
    /// staging that rhs in shared memory is rejected rather than adding boundary tracking to the
    /// normal staging path.
    pub fn normalized(self, _mask: TapMask, _guard: DivGuard) -> Tile<T> {
        unexpanded!()
    }
}

impl<T: Float> TileExpand<T> {
    pub fn __expand_normalized_method(
        mut self,
        scope: &Scope,
        mask: TapMask,
        guard: DivGuard,
    ) -> TileExpand<T> {
        validate_guard(guard);
        match &mut self.tile_kind {
            TileKindExpand::Procedural(data) => {
                assert!(
                    data.factor_count(scope).is_some(),
                    "Tile::normalized: the procedural recipe states no separable factorization"
                );
                data.normalization = Some((mask, guard));
            }
            TileKindExpand::Gmem(_)
            | TileKindExpand::Smem(_)
            | TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_) => {
                panic!("Tile::normalized: only a separable procedural tile has factor runs")
            }
        }
        self
    }
}

#[cube]
impl<T: Float> Tile<T> {
    /// Divide each cell by the matching denominator cell, broadcasting over axes the denominator
    /// omits. The walk follows the accumulator's existing spatial ownership, so it needs neither
    /// a cooperative cyclic scan nor synchronization.
    pub fn normalize<Meta: Numeric>(
        &mut self,
        denominator: &Tile<Meta>,
        #[comptime] guard: DivGuard,
    ) {
        comptime!(validate_guard(guard));
        let acc_space = comptime!(self.space.clone());
        let denominator_space = comptime!(denominator.space.clone());
        comptime!({
            assert!(
                denominator_space
                    .axes()
                    .all(|axis| acc_space.contains(axis)),
                "Tile::normalize: every denominator axis must be spanned by the accumulator"
            );
            for axis in denominator_space.axes() {
                assert_eq!(
                    denominator_space.extent_raw(axis),
                    acc_space.extent_raw(axis),
                    "Tile::normalize: a spanned denominator axis must match the accumulator; \
                     omit the axis to broadcast it"
                );
            }
        });

        match comptime!(acc_space.partitioner().clone()) {
            Partitioner::Final => {
                comptime!(assert!(
                    denominator_space.is_final(),
                    "Tile::normalize: the denominator cannot have levels below the accumulator"
                ));
                normalize_leaf(self, denominator, guard);
            }
            Partitioner::Level(_) => {
                let op_space = self.normalize_space(denominator);
                let unroll = self.tile_kind.static_level(comptime!(self.space.clone()));
                for region in Walk::over(op_space).with_unroll(unroll) {
                    let mut acc = self.at(&region);
                    if comptime!(denominator_space.is_final()) {
                        comptime!(assert!(
                            final_denominator_is_invariant(&self.space, &denominator.space),
                            "Tile::normalize: a denominator that is already final spans an axis \
                             still cut by the accumulator walk"
                        ));
                        acc.normalize(denominator, guard);
                    } else {
                        comptime!(assert!(
                            levels_align(&self.space, &denominator.space),
                            "Tile::normalize: accumulator and denominator levels must use the \
                             same edge on every denominator axis"
                        ));
                        let denominator = denominator.at(&region);
                        acc.normalize(&denominator, guard);
                    }
                }
            }
        }
    }

    fn normalize_space<Meta: Numeric>(&self, denominator: &Tile<Meta>) -> Space {
        witnessed_space(
            comptime!(self.space.clone()),
            self,
            denominator,
            denominator,
        )
    }
}

#[cube]
fn normalize_leaf<Acc: Float, Meta: Numeric>(
    acc: &mut Tile<Acc>,
    denominator: &Tile<Meta>,
    #[comptime] guard: DivGuard,
) {
    let space = comptime!(acc.space.clone());
    match &mut acc.tile_kind {
        TileKind::Gmem(data) | TileKind::Smem(data) => {
            comptime!(assert!(
                data.lane_share == LaneShare::Whole,
                "Tile::normalize: a folded memory accumulator is only a lane partial"
            ));
            comptime!(assert!(
                data.projection.is_direct() && !data.layout.projection.is_tiled(),
                "Tile::normalize: the accumulator needs a direct, untiled writable layout"
            ));
            normalize_memory(data, denominator, space, guard);
        }
        TileKind::PlaneTile(tile) => normalize_plane_tile(tile, denominator, space, guard),
        TileKind::PlanePartition(partition) => {
            comptime!(assert!(
                partition.m_tiles == 1 && partition.n_tiles == 1,
                "Tile::normalize: a multi-tile partition must be selected at its partition level"
            ));
            let mut tile = partition.at(0usize, 0usize);
            normalize_plane_tile(&mut tile, denominator, space, guard);
        }
        TileKind::TmaGmem(_) => panic!("Tile::normalize: a tma source is not writable"),
        TileKind::Procedural(_) => panic!("Tile::normalize: a procedural tile is not writable"),
    }
}

#[cube]
fn normalize_memory<Acc: Float, Meta: Numeric>(
    acc: &mut MemData<Acc>,
    denominator: &Tile<Meta>,
    #[comptime] space: Space,
    #[comptime] guard: DivGuard,
) {
    let width = comptime!(acc.store.vector_size);
    let size!(W) = width;
    let total = comptime!(space.tile_size());
    comptime!(assert!(
        total.is_multiple_of(width),
        "Tile::normalize: accumulator extent must contain whole store-width lines"
    ));
    let lines = comptime!(total / width);
    let mut values = acc.flat_mut::<W>();
    normalize_memory_lines::<Acc, Meta, W>(&mut values, denominator, space, lines, width, guard);
}

#[cube]
fn normalize_plane_tile<Acc: Float, Meta: Numeric>(
    tile: &mut PlaneTile<Acc>,
    denominator: &Tile<Meta>,
    #[comptime] space: Space,
    #[comptime] guard: DivGuard,
) {
    match tile {
        PlaneTile::Register(data) => {
            comptime!(assert!(
                data.lane_share == LaneShare::Whole,
                "Tile::normalize: a folded register accumulator is only a lane partial"
            ));
            let width = comptime!(data.vector_size);
            let lines = comptime!(data.mr * data.nr);
            comptime!(assert!(
                space.tile_size() == lines * width,
                "Tile::normalize: RegisterData shape mismatch with accumulator space"
            ));
            normalize_register_lines::<Acc, Meta>(data, denominator, space, lines, width, guard);
        }
        PlaneTile::Cmma(_) | PlaneTile::Mma(_) => {
            panic!("Tile::normalize: a hardware mma fragment has no addressable elementwise layout")
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn normalize_memory_lines<Acc: Float, Meta: Numeric, W: Size>(
    values: &mut FlatViewMut<'_, Vector<Acc, W>>,
    denominator: &Tile<Meta>,
    #[comptime] space: Space,
    #[comptime] lines: usize,
    #[comptime] width: usize,
    #[comptime] guard: DivGuard,
) {
    let size!(D) = denominator.vector_size();
    let denominator_view = denominator.nd_packed::<D>();
    let denominator_space = comptime!(denominator.space.clone());
    let extents = comptime!(
        (0..space.rank())
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    );
    for line in 0..lines {
        let current = values.read(line);
        values.write(
            line,
            normalize_line::<Acc, Meta, W, D>(
                current,
                &denominator_view,
                comptime!(space.clone()),
                comptime!(denominator_space.clone()),
                comptime!(extents.clone()),
                line,
                width,
                denominator.vector_size(),
                guard,
            ),
        );
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn normalize_register_lines<Acc: Float, Meta: Numeric>(
    values: &mut RegisterData<Acc>,
    denominator: &Tile<Meta>,
    #[comptime] space: Space,
    #[comptime] lines: usize,
    #[comptime] width: usize,
    #[comptime] guard: DivGuard,
) {
    let size!(D) = denominator.vector_size();
    let denominator_view = denominator.nd_packed::<D>();
    let denominator_space = comptime!(denominator.space.clone());
    let extents = comptime!(
        (0..space.rank())
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    );
    #[unroll]
    for line in 0..lines {
        values.data[line] = normalize_line::<Acc, Meta, RA, D>(
            values.data[line],
            &denominator_view,
            comptime!(space.clone()),
            comptime!(denominator_space.clone()),
            comptime!(extents.clone()),
            line,
            width,
            denominator.vector_size(),
            guard,
        );
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn normalize_line<Acc: Float, Meta: Numeric, W: Size, D: Size>(
    value: Vector<Acc, W>,
    denominator: &MaskedView<'_, Vector<Meta, D>, CoordsDyn>,
    #[comptime] acc_space: Space,
    #[comptime] denominator_space: Space,
    #[comptime] extents: Vec<usize>,
    line: usize,
    #[comptime] width: usize,
    #[comptime] denominator_width: usize,
    #[comptime] guard: DivGuard,
) -> Vector<Acc, W> {
    if comptime!(lane_aligned(
        &acc_space,
        &denominator_space,
        width,
        denominator_width
    )) {
        normalize_aligned::<Acc, Meta, W, D>(
            value,
            denominator,
            comptime!(acc_space.clone()),
            comptime!(denominator_space.clone()),
            comptime!(extents.clone()),
            line,
            width,
            guard,
        )
    } else if comptime!(lane_invariant(&acc_space, &denominator_space, width)) {
        normalize_invariant::<Acc, Meta, W, D>(
            value,
            denominator,
            comptime!(acc_space.clone()),
            comptime!(denominator_space.clone()),
            comptime!(extents.clone()),
            line,
            width,
            denominator_width,
            guard,
        )
    } else {
        normalize_general::<Acc, Meta, W, D>(
            value,
            denominator,
            comptime!(acc_space.clone()),
            comptime!(denominator_space.clone()),
            comptime!(extents.clone()),
            line,
            width,
            denominator_width,
            guard,
        )
    }
}

fn lane_aligned(acc: &Space, denominator: &Space, aw: usize, dw: usize) -> bool {
    let inner = acc.axis_at(acc.rank() - 1);
    aw == dw && denominator.contains(inner) && denominator.axis_at(denominator.rank() - 1) == inner
}

fn lane_invariant(acc: &Space, denominator: &Space, aw: usize) -> bool {
    aw == 1 || !denominator.contains(acc.axis_at(acc.rank() - 1))
}

/// A final denominator has no level left to window. It can pass through this accumulator level
/// only when every axis it spans has a single static region at the level.
fn final_denominator_is_invariant(acc: &Space, denominator: &Space) -> bool {
    acc.axes()
        .all(|axis| !denominator.contains(axis) || (!acc.is_dynamic(axis) && acc.count(axis) == 1))
}

/// Both operands consume a level from the same region. The denominator may omit broadcast axes,
/// but every axis it keeps must interpret that region coordinate at the accumulator's edge.
fn levels_align(acc: &Space, denominator: &Space) -> bool {
    denominator
        .axes()
        .all(|axis| acc.partitioner().edge(axis) == denominator.partitioner().edge(axis))
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn normalize_aligned<Acc: Float, Meta: Numeric, W: Size, D: Size>(
    value: Vector<Acc, W>,
    denominator: &MaskedView<'_, Vector<Meta, D>, CoordsDyn>,
    #[comptime] acc_space: Space,
    #[comptime] denominator_space: Space,
    #[comptime] extents: Vec<usize>,
    line: usize,
    #[comptime] width: usize,
    #[comptime] guard: DivGuard,
) -> Vector<Acc, W> {
    let acc_coords = unravel(
        &const_coords(comptime!(extents)),
        (line * comptime!(width)).fcast::<u32>(),
    );
    let denominator_coords = denominator_coords(
        comptime!(denominator_space),
        comptime!(acc_space),
        &acc_coords,
        width,
    );
    let divisor = denominator.read(denominator_coords);
    let epsilon = Acc::new(comptime!(guard.epsilon));
    let fallback = Acc::new(comptime!(guard.fallback));
    let mut safe = Vector::<Acc, W>::cast_from(Acc::from_int(1));
    #[unroll]
    for lane in 0..width {
        let d = Acc::cast_from(divisor.extract(comptime!(lane)));
        safe.insert(
            comptime!(lane),
            select(d.abs() > epsilon, d, Acc::from_int(1)),
        );
    }
    // The aligned path keeps the native line operation: one vector divide, then only the guarded
    // lanes are substituted with the requested fallback multiplier.
    let quotient = value / safe;
    let mut result = quotient;
    #[unroll]
    for lane in 0..width {
        let d = Acc::cast_from(divisor.extract(comptime!(lane)));
        result.insert(
            comptime!(lane),
            select(
                d.abs() > epsilon,
                quotient.extract(comptime!(lane)),
                value.extract(comptime!(lane)) * fallback,
            ),
        );
    }
    result
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn normalize_invariant<Acc: Float, Meta: Numeric, W: Size, D: Size>(
    value: Vector<Acc, W>,
    denominator: &MaskedView<'_, Vector<Meta, D>, CoordsDyn>,
    #[comptime] acc_space: Space,
    #[comptime] denominator_space: Space,
    #[comptime] extents: Vec<usize>,
    line: usize,
    #[comptime] width: usize,
    #[comptime] denominator_width: usize,
    #[comptime] guard: DivGuard,
) -> Vector<Acc, W> {
    let acc_coords = unravel(
        &const_coords(comptime!(extents)),
        (line * comptime!(width)).fcast::<u32>(),
    );
    let divisor = denominator_at::<Acc, Meta, D>(
        denominator,
        denominator_space,
        acc_space,
        &acc_coords,
        denominator_width,
    );
    value * Vector::<Acc, W>::cast_from(guarded_recip::<Acc>(divisor, guard))
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn normalize_general<Acc: Float, Meta: Numeric, W: Size, D: Size>(
    value: Vector<Acc, W>,
    denominator: &MaskedView<'_, Vector<Meta, D>, CoordsDyn>,
    #[comptime] acc_space: Space,
    #[comptime] denominator_space: Space,
    #[comptime] extents: Vec<usize>,
    line: usize,
    #[comptime] width: usize,
    #[comptime] denominator_width: usize,
    #[comptime] guard: DivGuard,
) -> Vector<Acc, W> {
    let mut result = value;
    #[unroll]
    for lane in 0..width {
        let acc_coords = unravel(
            &const_coords(comptime!(extents.clone())),
            (line * comptime!(width) + comptime!(lane)).fcast::<u32>(),
        );
        let divisor = denominator_at::<Acc, Meta, D>(
            denominator,
            comptime!(denominator_space.clone()),
            comptime!(acc_space.clone()),
            &acc_coords,
            denominator_width,
        );
        result.insert(
            comptime!(lane),
            guarded_div::<Acc>(value.extract(comptime!(lane)), divisor, guard),
        );
    }
    result
}

#[cube]
fn denominator_at<Acc: Float, Meta: Numeric, D: Size>(
    denominator: &MaskedView<'_, Vector<Meta, D>, CoordsDyn>,
    #[comptime] denominator_space: Space,
    #[comptime] acc_space: Space,
    acc_coords: &Coords<u32>,
    #[comptime] width: usize,
) -> Acc {
    let coords = denominator_coords(
        comptime!(denominator_space.clone()),
        comptime!(acc_space.clone()),
        acc_coords,
        width,
    );
    let line = denominator.read(coords);
    let lane = if comptime!(width == 1) {
        0usize.runtime()
    } else {
        let axis = comptime!(denominator_space.axis_at(denominator_space.rank() - 1));
        let p = comptime!(acc_space.position(axis));
        acc_coords
            .at(p)
            .frem(comptime!(width as u32))
            .fcast::<usize>()
    };
    Acc::cast_from(line.extract_dynamic(lane))
}

#[cube]
fn denominator_coords(
    #[comptime] denominator_space: Space,
    #[comptime] acc_space: Space,
    acc_coords: &Coords<u32>,
    #[comptime] width: usize,
) -> CoordsDyn {
    resolve_nd_coords(
        denominator_space,
        acc_space,
        comptime!(Vec::new()),
        acc_coords,
        &Coords::<u32>::new(),
        width,
        true,
    )
}

#[cube]
fn guarded_div<E: Float>(value: E, denominator: E, #[comptime] guard: DivGuard) -> E {
    let epsilon = E::new(comptime!(guard.epsilon));
    let fallback = E::new(comptime!(guard.fallback));
    let valid = denominator.abs() > epsilon;
    let safe = select(valid, denominator, E::from_int(1));
    select(valid, value / safe, value * fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_guard_accepts_finite_non_negative_thresholds() {
        validate_guard(DivGuard::default());
        validate_guard(DivGuard {
            epsilon: 1.0e-7,
            fallback: -1.0,
        });
    }

    #[test]
    fn division_guard_rejects_invalid_thresholds() {
        for epsilon in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| validate_guard(DivGuard {
                    epsilon,
                    fallback: 0.0,
                }))
                .is_err(),
                "epsilon {epsilon:?} should be rejected"
            );
        }
    }
}
