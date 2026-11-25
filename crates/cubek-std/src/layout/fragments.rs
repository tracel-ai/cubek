//! A buffer's physical dims grouped by the logical dim each is a fragment of: what every reader of
//! a layout reads, a plain buffer's one fragment a dim and a tiled one's several alike.

use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, SmallVec, Strides, Tiling};

/// One logical dim's fragments, coarsest first: `(extent, stride)` in values.
pub type Fragments = SmallVec<[(usize, usize); 4]>;

/// A tiled buffer's physical dims, grouped by the logical dim each is a fragment of, coarsest
/// first: `(extent, stride)` in values.
pub fn fragments(meta: &Metadata) -> SmallVec<[Fragments; 4]> {
    let rank = meta
        .logical_rank()
        .expect("a buffer's tiling describes its own rank");
    let fragment = |&physical: &usize| (meta.shape[physical], meta.strides[physical]);
    meta.tiling
        .physical_dims(rank)
        .iter()
        .map(|dims| dims.iter().map(fragment).collect())
        .collect()
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

/// The one matrix the last two logical dims of `meta` hold, each stored as its [`fragments`] say;
/// `None` where a leading dim holds more than one, or there are not two dims.
pub fn one_matrix(meta: &Metadata) -> Option<Metadata> {
    let dims = fragments(meta);
    let rank = dims.len();
    let held = |dim: &[(usize, usize)]| dim.iter().map(|&(extent, _)| extent).product::<usize>();
    (rank >= 2 && dims[..rank - 2].iter().all(|dim| held(dim) == 1))
        .then(|| assemble(&[&dims[rank - 2], &dims[rank - 1]]))
}

#[cfg(test)]
mod tests {
    use super::{fragments_of, one_matrix};
    use cubecl::zspace::metadata::Metadata;
    use cubecl::zspace::{Shape, Strides, Tiling};

    /// Leading dims of one hold a single matrix, read at its own strides; a leading dim of more
    /// holds several, and one dim holds no matrix. A matrix stored in tiles keeps its tiles.
    #[test]
    fn only_leading_dims_of_one_hold_one_matrix() {
        let (k, n) = (64, 48);
        let plain =
            |shape: &[usize], strides: &[usize]| Metadata::new(shape.to_vec(), strides.to_vec());
        assert_eq!(
            one_matrix(&plain(&[k, n], &[n, 1])),
            Some(plain(&[k, n], &[n, 1]))
        );
        assert_eq!(
            one_matrix(&plain(&[1, k, n], &[k * n, 1, k])),
            Some(plain(&[k, n], &[1, k]))
        );
        assert_eq!(one_matrix(&plain(&[2, k, n], &[k * n, n, 1])), None);
        assert_eq!(one_matrix(&plain(&[k], &[1])), None);
        let tiles = Metadata::new(
            Shape::from(vec![k / 8, n / 8, 8, 8]),
            Strides::from(vec![n * 8, 64, 8, 1]),
        )
        .with_tiling(Tiling::new(&[2, 2]).expect("two fragments a dim"))
        .expect("the tiling describes the dims");
        assert_eq!(one_matrix(&tiles), Some(tiles.clone()));
    }

    /// `[m, k]` stored `[m/16, k/16, m%8, k/2%4, k/8%2, m/8%2, k%2]`, as a tensor core takes a tile
    /// of sixteen by sixteen.
    #[test]
    fn fragments_are_grouped_whatever_dims_they_are_listed_as() {
        let meta = Metadata::new(
            Shape::from(vec![3, 5, 8, 4, 2, 2, 2]),
            Strides::from(vec![1280, 256, 32, 8, 4, 2, 1]),
        )
        .with_tiling(
            Tiling::from_physical_dims(&[&[0, 5, 2][..], &[1, 4, 3, 6]])
                .expect("each dim is one fragment"),
        )
        .expect("the tiling describes the dims");

        let [m, k] = fragments_of::<2>(&meta).expect("a matrix");
        assert_eq!(m.as_slice(), &[(3, 1280), (2, 2), (8, 32)]);
        assert_eq!(k.as_slice(), &[(5, 256), (2, 4), (4, 8), (2, 1)]);
    }
}
