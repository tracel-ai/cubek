//! [`MatrixBinding`]: a matrix operand's binding, asked how it is stored.

use cubecl::{
    prelude::TensorBinding,
    zspace::{Tiling, metadata::Metadata},
};
use cubek_tile::{Axis, Geometry, Level, Partitioning, Space, layout::StorageTiling};

use crate::{definition::MatmulSetupError, tiled::labels};

/// A matrix operand's binding: leading batch dims, then the two dims of the matrix, stored in rows
/// or in storage tiles. A routine asks it how it is stored rather than reading the binding's dims
/// itself, so every routine reads a tiled operand the same way. `name` says which operand a
/// refusal is about.
pub(crate) struct MatrixBinding<'a> {
    binding: &'a TensorBinding,
    name: &'a str,
}

impl<'a> MatrixBinding<'a> {
    /// The matrix's rows, as its storage is read.
    pub(crate) const ROWS: Axis = Axis(0);
    /// The matrix's columns.
    pub(crate) const COLS: Axis = Axis(1);

    pub(crate) fn new(binding: &'a TensorBinding, name: &'a str) -> Self {
        Self { binding, name }
    }

    /// Its logical `(batches, rows, cols)`: the tensor states how it is stored, so its own metadata
    /// folds the pieces back and no routine has to be told.
    pub(crate) fn dims(&self) -> (Vec<usize>, usize, usize) {
        let logical = Metadata::new(self.binding.shape.clone(), self.binding.strides.clone())
            .with_tiling(self.binding.tiling)
            .expect("a binding's tiling describes its own rank")
            .logical_shape()
            .expect("a binding's tiling describes its own rank");
        let dims = logical.as_slice();
        let split = dims.len() - 2;
        (dims[..split].to_vec(), dims[split], dims[split + 1])
    }

    /// The labels of its trailing dims: [`ROWS`](Self::ROWS) and [`COLS`](Self::COLS), or one per
    /// piece of a storage-tiled one, in the order its tiling lists them.
    pub(crate) fn labels(&self) -> Vec<Axis> {
        match self.binding.tiling.is_tiled() {
            true => StorageTiling::stored(self.binding.tiling, 2, self.binding.shape.len())
                .order(&[Self::ROWS, Self::COLS]),
            false => vec![Self::ROWS, Self::COLS],
        }
    }

    /// Its outermost storage tile, `(rows, cols)`, whatever finer pieces it holds and in whatever
    /// order: each matrix dim over its grid's count, the grid being each dim's coarsest piece,
    /// which a tiling lists first. `None` in rows.
    ///
    /// # Errors
    ///
    /// A batch dim stored in pieces, which no routine reads.
    #[allow(clippy::result_large_err)]
    pub(crate) fn tile(&self) -> Result<Option<(usize, usize)>, MatmulSetupError> {
        if !self.binding.tiling.is_tiled() {
            return Ok(None);
        }
        let rank = self.binding.shape.len();
        let logical_rank = self
            .binding
            .tiling
            .logical_rank(rank)
            .map_err(|e| self.refused(format!("{e:?}")))?;
        let fragments = self.binding.tiling.fragments(logical_rank);
        let batches = logical_rank - 2;
        if fragments[..batches].iter().any(|&n| n != 1) {
            return Err(self.refused(format!(
                "batch dims are stored plain, got {fragments:?} pieces per dim"
            )));
        }
        let (_, rows, cols) = self.dims();
        Ok(Some((
            rows / self.binding.shape[batches],
            cols / self.binding.shape[batches + 1],
        )))
    }

    /// Its outermost storage tile, when that tile is stored a row at a time: the pieces under it
    /// may be finer than a routine reads (a vector read, a packed word), and it is read the same
    /// as long as they fuse back into it a row at a time: the buffer
    /// [serves](Geometry::serves) a row-first tile. The order the tiles themselves follow one
    /// another in is the strides', whatever it is.
    ///
    /// # Errors
    ///
    /// What [`tile`](Self::tile) refuses, or a tile that is not stored a row at a time.
    #[allow(clippy::result_large_err)]
    pub(crate) fn row_first_tile(&self) -> Result<Option<(usize, usize)>, MatmulSetupError> {
        let Some((rows, cols)) = self.tile()? else {
            return Ok(None);
        };
        let row_first = [(Self::COLS, cols), (Self::ROWS, rows)];
        Geometry::from(self.binding)
            .serves(&row_first, &self.labels())
            .map_err(|why| {
                self.refused(format!(
                    "it is not stored in {rows}x{cols} tiles a row at a time: {why}"
                ))
            })?;
        Ok(Some((rows, cols)))
    }

    /// The binding as a routine that moves whole tiles sees it: the same buffer, its tile's pieces
    /// fused into one row-first `rows x cols` tile, `[.., R/tr, C/tc, tr, tc]` over its grid's own
    /// strides. In rows, the binding as it is.
    ///
    /// # Errors
    ///
    /// What [`row_first_tile`](Self::row_first_tile) refuses.
    #[allow(clippy::result_large_err)]
    pub(crate) fn fused(&self) -> Result<TensorBinding, MatmulSetupError> {
        let Some((tr, tc)) = self.row_first_tile()? else {
            return Ok(self.binding.clone());
        };
        let (batches, _, _) = self.dims();
        let batches = batches.len();
        let mut shape: Vec<usize> = self.binding.shape[..batches + 2].to_vec();
        let mut strides: Vec<usize> = self.binding.strides[..batches + 2].to_vec();
        shape.extend([tr, tc]);
        strides.extend([tc, 1]);
        let fragments: Vec<usize> = (0..batches).map(|_| 1).chain([2, 2]).collect();
        let mut fused = self.binding.clone();
        fused.shape = shape.into();
        fused.strides = strides.into();
        fused.tiling = Tiling::new(&fragments).expect("one level over two dims fits any tiling");
        Ok(fused)
    }

    /// Refuse, on the host, a storage tile that is the tile of none of `levels` on
    /// `(rows, cols)`: the launch matches it on a worker thread, where a refusal reaches no
    /// caller. In rows, it passes.
    ///
    /// # Errors
    ///
    /// What [`row_first_tile`](Self::row_first_tile) refuses, or a tile no level cuts to.
    #[allow(clippy::result_large_err)]
    pub(crate) fn cut_by(
        &self,
        space: &Space,
        levels: &[Level],
        (rows, cols): (Axis, Axis),
    ) -> Result<(), MatmulSetupError> {
        let Some(tile) = self.row_first_tile()? else {
            return Ok(());
        };
        let cuts_to = |i: usize| {
            let leaf = Partitioning::new(space.clone(), levels[..=i].to_vec()).leaf();
            (leaf.extent(rows), leaf.extent(cols)) == tile
        };
        if (0..levels.len()).any(cuts_to) {
            return Ok(());
        }
        let partitioning = Partitioning::new(space.clone(), levels.to_vec());
        Err(self.refused(format!(
            "it is stored in {tile:?} storage tiles, which is the tile of no level of this \
             routine's nest; tile the tensor to one of its tiles\n\n{}",
            partitioning.table(&labels(space))
        )))
    }

    /// Refuse a binding a routine that addresses each window by a row stride off a scalar offset
    /// cannot read: one whose innermost dim does not step by one. A storage-tiled one stores each
    /// tile a row at a time; one stored a column at a time is refused, however its grid is
    /// ordered.
    ///
    /// # Errors
    ///
    /// Its innermost dim strides past one.
    #[allow(clippy::result_large_err)]
    pub(crate) fn row_major(&self) -> Result<(), MatmulSetupError> {
        if self.binding.strides.last() == Some(&1) {
            return Ok(());
        }
        Err(self.refused(
            match self.binding.tiling.is_tiled() {
                true => "cannot be read, its storage tiles are not stored a row at a time",
                false => "cannot be read, it is not row-major contiguous",
            }
            .to_string(),
        ))
    }

    /// Refuse a storage tile that is not `stage`: a routine that stages whole storage tiles
    /// stages the tile the tensor is stored in. In rows, it passes.
    ///
    /// # Errors
    ///
    /// What [`row_first_tile`](Self::row_first_tile) refuses, or a tile other than `stage`.
    #[allow(clippy::result_large_err)]
    pub(crate) fn stages(&self, stage: (usize, usize)) -> Result<(), MatmulSetupError> {
        match self.row_first_tile()? {
            Some(tile) if tile != stage => Err(self.refused(format!(
                "it is stored in {tile:?} storage tiles but the plan stages {stage:?}; the storage \
                 tile names the stage, so tile the tensor to the plan's stage or plan for the tile"
            ))),
            _ => Ok(()),
        }
    }

    fn refused(&self, why: String) -> MatmulSetupError {
        MatmulSetupError::InvalidConfig(Box::new(format!("{}: {why}", self.name)))
    }
}

#[cfg(test)]
mod tests {
    use cubecl::{prelude::TensorBinding, zspace::Tiling};

    use super::*;
    use crate::tiled::{K, M, N};

    fn binding(shape: &[usize], tiling: Tiling) -> TensorBinding {
        let client = cubecl::test_device().client();
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let len: usize = shape.iter().product();
        TensorBinding {
            handle: client.empty(len * 4).binding(),
            strides: strides.into(),
            shape: shape.to_vec().into(),
            tiling,
        }
    }

    #[test]
    /// A storage tile is the tile of one of a routine's levels or it is refused on the host,
    /// where a caller hears it: matched on a worker thread, a refusal reaches no one.
    fn a_stored_tile_that_is_a_level_passes_and_one_that_is_none_is_refused() {
        let space = Space::new(&[(M, 64), (N, 64), (K, 32)]);
        let levels = cubek_tile::Levels::leaf(&[(M, 8), (N, 8), (K, 4)])
            .walk_every(&[K])
            .planes(&[(M, 2), (N, 4)])
            .cubes(&[M, N])
            .build();
        let cube_tile = binding(&[2, 4, 16, 32], Tiling::new(&[2, 2]).unwrap());
        MatrixBinding::new(&cube_tile, "out")
            .cut_by(&space, &levels, (M, N))
            .unwrap();
        let no_level = binding(&[4, 4, 16, 16], Tiling::new(&[2, 2]).unwrap());
        assert!(
            MatrixBinding::new(&no_level, "out")
                .cut_by(&space, &levels, (M, N))
                .is_err()
        );
        let plain = binding(&[64, 64], Tiling::UNTILED);
        MatrixBinding::new(&plain, "out")
            .cut_by(&space, &levels, (M, N))
            .unwrap();
    }
}
