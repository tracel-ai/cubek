//! The runtime side of a [`Projection`]: [`RuntimeMap`], the values a mapping only knows in the
//! kernel, and the folds that drive a buffer's physical shape and strides against the digits an
//! operand's coordinates decompose into. Free `#[cube]` functions, not methods, because
//! `Projection` stays comptime-only (never a [`CubeType`]), like every other comptime/runtime mix
//! in this crate.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Axis, Coords, Fold, FoldExpand, FoldSeq, FoldSeqExpand, Projection, const_coords};

/// What a [`Projection`] cannot state at comptime: the values its [`Dynamic`](crate::Scale)
/// coefficients and divisors carry, and the phase its window origin sits at under a
/// [rational](Divisor) mapping. The two travel together because they are read together, per
/// physical axis, in one expression: [`AxisProjection`](crate::AxisProjection) resolves a tap
/// through the coefficients and off the phase at once, and a descent
/// ([`MemData::at`](crate::MemData)) advances both.
///
/// No coefficients and an all-zero phase is the whole of it for a fully-`Static` integer mapping,
/// which is every operand but a runtime-strided or fractionally scaled gather; [`Fold`] passes
/// that through, so carrying it costs nothing where it says nothing.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RuntimeMap {
    /// One value per [`Scale::Dynamic`](crate::Scale) coefficient in
    /// [`Projection::dynamic_scale_index`] order, plus one per [`Divisor::Dynamic`] axis in
    /// [`Projection::dynamic_divisor_index`] order, the two interleaved physical axis major (an
    /// axis's divisor follows its own coefficients). Named apart from the quantization scales in
    /// [`Store`](crate::Store), which are a different thing entirely.
    pub(crate) coefficients: Coords<u32>,
    /// The window origin's phase within its divisor, one per physical axis, in `0..divisor`.
    /// Constant `0` for every integer mapping, where the origin absorbs the offset whole.
    pub(crate) residues: Coords<u32>,
}

#[cube]
impl RuntimeMap {
    /// The runtime map an integer mapping carries: nothing to carry, and no phase on any of
    /// `physical_rank` axes.
    pub(crate) fn integral(#[comptime] physical_rank: usize) -> RuntimeMap {
        RuntimeMap {
            coefficients: Coords::<u32>::new(),
            residues: const_coords(comptime!(vec![0; physical_rank])),
        }
    }

    /// Materialize this map in mutable, per-slot kernel registers. A normal clone preserves the
    /// source expressions; a staged slot instead needs values that survive independently while a
    /// sibling slot is refilled for another region.
    pub(crate) fn stored(&self) -> RuntimeMap {
        RuntimeMap {
            coefficients: self.coefficients.stored(),
            residues: self.residues.stored(),
        }
    }

    /// Store a source window's complete runtime addressing state in this slot.
    pub(crate) fn store_from(&mut self, src: &RuntimeMap) {
        self.coefficients.store_from(&src.coefficients);
        self.residues.store_from(&src.residues);
    }
}

/// The logical extent per axis, folded from `projection`'s physical shape: a single-carrier axis
/// passes its physical extent through, a storage-tiled one multiplies its fragments' extents back
/// together. Reduces to `physical_shape` itself for an untiled projection. A free function, not a
/// method: `Projection` stays comptime-only (never a [`CubeType`]), so the runtime folding it
/// drives lives in an ordinary `#[cube]` function, like every other comptime/runtime mix in this
/// crate ([`top_window`](crate::GmemLayout)).
#[cube]
pub fn logical_extent(
    #[comptime] projection: Projection,
    physical_shape: &Coords<u32>,
) -> Coords<u32> {
    let mut bound = Coords::<u32>::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let picks = comptime!(projection.carriers(projection.logical_axes()[i]).to_vec());
        bound.push(physical_shape.fproduct(picks));
    }
    bound
}

/// The line offset one `edge`-sized tile step along `axis` moves under `projection`: `axis`'s
/// digits taken of `edge` itself rather than of a coordinate, dotted with `strides`. Exact because
/// one edge-step's offset is linear in the tile index: every edge divides its enclosing block, so
/// decomposing the step size the same way as a coordinate reconstructs the same advance. The
/// radices come from `physical_shape`, so a static store folds the whole thing to a constant and a
/// runtime-shaped one stays exact.
#[cube]
pub fn step_offset(
    #[comptime] projection: Projection,
    #[comptime] axis: Axis,
    #[comptime] edge: usize,
    physical_shape: &Coords<u32>,
    strides: &Coords<u32>,
) -> u32 {
    let carriers = comptime!(projection.carriers(axis));
    let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
    let mut parts = Sequence::<u32>::new();
    #[unroll]
    for k in 0..comptime!(carriers.len()) {
        let pa = comptime!(carriers[k]);
        let (finer, modulo) = comptime!(projection.digit(pa, axis));
        let scale = comptime!(projection.scale(pa, axis) as u32);
        let quot = comptime!(edge as u32)
            .runtime()
            .fdiv(physical_shape.fproduct(comptime!(finer.to_vec())));
        let digit = match comptime!(modulo) {
            Some(m) => quot.frem(physical_shape.at(m)),
            None => quot,
        };
        parts.push(digit.fmul(comptime!(scale).runtime()).fmul(strides.at(pa)));
    }
    parts.fsum(picks)
}

/// The inverse of `GmemLayout`'s `to_source_pos`: the logical
/// coordinate under `projection` that produced physical digits `digits` (one entry per physical
/// axis, already decoded off the flat physical index). Requires [`Projection::is_invertible`]; a
/// gathered (affine, scale != 1) projection never reaches this: the only projection a
/// [`GmemLayout`](crate::GmemLayout) carries is its buffer's own positional map, and a gathered
/// operand is either resolved a layer above it or staged through its own compacted [`Projection`],
/// never folded here directly.
#[cube]
pub fn fold_physical(
    #[comptime] projection: Projection,
    digits: &Coords<u32>,
    physical_shape: &Coords<u32>,
) -> CoordsDyn {
    comptime!(assert!(
        projection.is_invertible(),
        "Projection::fold_physical: not invertible (an affine term mixes several logical \
         coordinates into one physical cell)"
    ));
    let mut out = CoordsDyn::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let axis = comptime!(projection.logical_axes()[i]);
        let carriers = comptime!(projection.carriers(axis));
        let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
        let mut parts = Sequence::<u32>::new();
        #[unroll]
        for k in 0..comptime!(carriers.len()) {
            let pa = comptime!(carriers[k]);
            // Each digit weighs the block it sits above, the same finer extents that stripped it.
            let (finer, _) = comptime!(projection.digit(pa, axis));
            parts.push(
                digits
                    .at(pa)
                    .fmul(physical_shape.fproduct(comptime!(finer.to_vec()))),
            );
        }
        out.push(parts.fsum(picks));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);

    /// Folding `Σ digit * (extents it stripped)` back reconstructs the coordinate the digits were
    /// decomposed from, which is what `fold_physical` does to `to_source_pos`. Both are `#[cube]`,
    /// so the round trip is checked here on the digit positions they are built from.
    #[test]
    fn fold_physical_digits_invert_to_source_pos_digits() {
        use crate::PhysicalAxis;

        let shape = [3usize, 8];
        let layout = crate::ConcreteLayout::new(&[
            PhysicalAxis::new(A, shape[0]),
            PhysicalAxis::new(A, shape[1]),
        ]);
        let p = Projection::of_layout(&layout);
        assert!(p.is_invertible());
        let block = |pa| {
            p.digit(pa, A)
                .0
                .iter()
                .map(|&q| shape[q])
                .product::<usize>()
        };

        for coord in [0usize, 1, 7, 8, 9, 19, 23] {
            let digit = |pa| match p.digit(pa, A).1 {
                Some(m) => (coord / block(pa)) % shape[m],
                None => coord / block(pa),
            };
            assert_eq!(digit(0) * block(0) + digit(1) * block(1), coord);
        }
    }
}
