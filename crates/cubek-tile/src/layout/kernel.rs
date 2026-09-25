//! The runtime side of a [`Projection`]: [`RuntimeMap`], the values a mapping only knows in the
//! kernel, and the folds that drive a buffer's shape and strides against the digits an operand's
//! coordinates decompose into. Free `#[cube]` functions, as `Projection` is never a [`CubeType`].

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Axis, Coords, Integer, IntegerExpand, IntegerSeq, IntegerSeqExpand, Projection};

/// What a [`Projection`] cannot state at comptime: the values its [`Dynamic`](crate::Scale)
/// coefficients and divisors carry, and the phase its window origin sits at under a
/// [rational](Divisor) mapping.
///
/// They travel together because they are read together, per physical axis, in one expression:
/// [`ProjectionInKernel`](crate::ProjectionInKernel) resolves a tap through the coefficients and off the
/// phase at once, and a descent ([`Memory::at`](crate::Memory)) advances both.
///
/// No coefficients and an all-zero phase is the whole of it for a fully-`Static` integer mapping,
/// which is every operand but a runtime-strided or fractionally scaled gather; [`Integer`] passes
/// that through, so carrying it costs nothing where it says nothing.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RuntimeMap {
    /// One per [`Scale::Dynamic`](crate::Scale) coefficient ([`Projection::dynamic_scale_index`]
    /// order) and one per [`Divisor::Dynamic`] axis ([`Projection::dynamic_divisor_index`] order),
    /// interleaved physical axis major (divisor last); unlike [`Store`](crate::Store)'s scales.
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
            residues: Coords::constant(comptime!(vec![0; physical_rank])),
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
/// together, or `physical_shape` itself when untiled. Free since `Projection` is no [`CubeType`].
#[cube]
pub(crate) fn logical_extent(
    #[comptime] projection: Projection,
    physical_shape: &Coords<u32>,
) -> Coords<u32> {
    let mut bound = Coords::<u32>::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let picks = comptime!(projection.carriers(projection.logical_axes()[i]).to_vec());
        bound.push(physical_shape.product(picks));
    }
    bound
}

/// The line offset one `edge`-sized tile step along `axis` moves under `projection`: `axis`'s
/// digits taken of `edge` itself rather than of a coordinate, dotted with `strides`. Radices come
/// from `physical_shape`: a static store folds to a constant, a runtime-shaped one stays exact.
///
/// Exact because one edge-step's offset is linear in the tile index: every edge divides its
/// enclosing block, so decomposing the step size like a coordinate reconstructs the same advance.
#[cube]
pub(crate) fn step_offset(
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
            .divided_by(physical_shape.product(comptime!(finer.to_vec())));
        let digit = match comptime!(modulo) {
            Some(m) => quot.remainder(physical_shape.at(m)),
            None => quot,
        };
        parts.push(
            digit
                .times(comptime!(scale).runtime())
                .times(strides.at(pa)),
        );
    }
    parts.sum(picks)
}

/// The inverse of `BufferLayout`'s `to_source_pos`: the logical coordinate under `projection` that
/// produced physical digits `digits` (one per physical axis, already decoded off the flat index).
/// Requires [`Projection::is_invertible`]: a gathered (affine, scale != 1) one never reaches this.
///
/// A [`BufferLayout`](crate::BufferLayout) only carries its buffer's own positional map: a gathered
/// operand is resolved a layer above it or staged through its own compacted [`Projection`].
#[cube]
pub(crate) fn fold_physical(
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
                    .times(physical_shape.product(comptime!(finer.to_vec()))),
            );
        }
        out.push(parts.sum(picks));
    }
    out
}

/// Digit `j` of flat line `x` under `shape`'s row-major suffix strides.
#[cube]
pub(crate) fn line_digit(x: u32, shape: &Coords<u32>, #[comptime] j: usize) -> u32 {
    let plen = shape.len();
    x.divided_by(shape.product(comptime!(((j + 1)..plen).collect::<Vec<_>>())))
        .remainder(shape.at(j))
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
        let shape = [3usize, 8];
        let p = Projection::new(
            &[A],
            &[crate::PhysicalAxisMap::of(A), crate::PhysicalAxisMap::of(A)],
        );
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
