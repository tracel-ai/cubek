//! A buffer's physical dims grouped by the logical dim each is a fragment of: what every reader of
//! a layout reads, a plain buffer's one fragment a dim and a tiled one's several alike.

use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, SmallVec, Strides, Tiling};

/// One logical dim's fragments, coarsest first: `(extent, stride)` in values.
pub type Fragments = SmallVec<[(usize, usize); 4]>;

/// A tiled buffer's physical dims, grouped by the logical dim each is a fragment of, coarsest
/// first: `(extent, stride)` in values. Physical dims are level-major in logical order: every
/// dim's coarsest fragment, then every dim still deep enough gives its next.
pub fn fragments(meta: &Metadata) -> SmallVec<[Fragments; 4]> {
    let rank = meta
        .logical_rank()
        .expect("a buffer's tiling describes its own rank");
    let counts = meta.tiling.fragments(rank);
    let mut dims: SmallVec<[Fragments; 4]> = SmallVec::from_elem(Fragments::new(), rank);
    let mut physical = 0;
    for level in 0..counts.iter().copied().max().unwrap_or(0) {
        for (dim, &count) in counts.iter().enumerate() {
            if level < count {
                dims[dim].push((meta.shape[physical], meta.strides[physical]));
                physical += 1;
            }
        }
    }
    dims
}

/// [`fragments`] of a buffer of `D` logical dims, one entry a dim; `Err` holds the rank it has
/// instead.
pub fn fragments_of<const D: usize>(meta: &Metadata) -> Result<[Fragments; D], usize> {
    let dims = fragments(meta);
    match dims.len() == D {
        true => Ok(core::array::from_fn(|dim| dims[dim].clone())),
        false => Err(dims.len()),
    }
}

/// The metadata whose logical dim `i` is stored as `dims[i]`, level-major.
pub fn assemble(dims: &[&[(usize, usize)]]) -> Metadata {
    let deepest = dims.iter().map(|dim| dim.len()).max().unwrap_or(0);
    let mut shape = Vec::new();
    let mut strides = Vec::new();
    for level in 0..deepest {
        for fragments in dims {
            if let Some(&(extent, stride)) = fragments.get(level) {
                shape.push(extent);
                strides.push(stride);
            }
        }
    }
    let counts: SmallVec<[usize; 4]> = dims.iter().map(|dim| dim.len()).collect();
    Metadata::new(Shape::from(shape), Strides::from(strides))
        .with_tiling(Tiling::new(&counts).expect("tile order's fragments fit a tiling"))
        .expect("the tiling describes the dims assembled")
}

/// The tile the trailing two logical dims of `meta` are stored in, one extent per dim in the order
/// it states them: every fragment of a dim but its coarsest, which counts the tiles. A buffer of
/// one fragment a dim is stored in tiles of one value, `(1, 1)`. `(k, n)` for a weight's own
/// statement, `(n, k)` for its rows.
pub fn tile(meta: &Metadata) -> (usize, usize) {
    let dims = fragments(meta);
    let inside =
        |dim: &Fragments| -> usize { dim.iter().skip(1).map(|&(extent, _)| extent).product() };
    let [first, second] = [&dims[dims.len() - 2], &dims[dims.len() - 1]];
    (inside(first), inside(second))
}
