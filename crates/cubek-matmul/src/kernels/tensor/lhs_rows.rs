use crate::kernels::tensor::{Placement, PlainTensor, StoredTensor};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{Bound, Decline, Delivery};
use cubecl::tile::{Launcher, TileTensorLaunch};
use cubek_microkernel::row_store::{ROWS, RowStore, WIDTH};
use cubek_std::layout::stored::{BlockScales, OuterScale};
use cubek_std::matmul_dimensions::{K, M};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable};

/// What a tiled launch declines before preparing anything: an output that is not a float matrix
/// of the producer's rows stored in tiles, at the producer's element.
pub const TILED_ROWS_UNSERVED: Decline = Decline::Unserved {
    why: "a tiled output is a float matrix of the producer's rows, stored in tiles, at its element",
};

/// What a packed launch declines before preparing anything: an output that is not a product's lhs
/// (`[M, K]`) holding the producer's rows.
pub const PACKED_ROWS_UNSERVED: Decline = Decline::Unserved {
    why: "a packed output is a product's [M, K] lhs, the producer's rows by their width",
};

/// What a producer declines of an output of 8-bit floats under scales constant along the
/// contraction: no producer writes one yet.
pub const SCALED_ROWS_UNSERVED: Decline = Decline::Unserved {
    why: "no producer writes its rows as 8-bit floats under scales constant along the contraction",
};

/// A float `[rows, width]` matrix stored in tiles that a producer writes its rows into, as a
/// product reads an operand stored in tiles: allocated by the caller, and bound over the rows and
/// the row span with the tiling its binding states, which every store's write lands through.
/// Nothing is staged and nothing is relaid out after.
#[derive(Debug, Clone)]
pub struct TiledRows {
    tensor: PlainTensor,
}

impl TiledRows {
    /// `out`, a matrix of `rows` rows of `width` elements of `element` stored in tiles.
    ///
    /// # Errors
    ///
    /// [`TILED_ROWS_UNSERVED`].
    pub fn new(
        out: PlainTensor,
        rows: usize,
        width: usize,
        element: ElemType,
    ) -> Result<Self, Decline> {
        let fits = out.is_tiled() && out.elem == element && out.shape().as_slice() == [rows, width];
        match fits {
            true => Ok(Self { tensor: out }),
            false => Err(TILED_ROWS_UNSERVED),
        }
    }

    /// The matrix bound under `launcher` over the rows and the row span, `width` elements a line.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses to bind the matrix the routine prepared.
    pub fn bind(&self, launcher: &Launcher, width: usize) -> Result<Bound, Decline> {
        launcher
            .arg(self.tensor.binding.clone(), self.tensor.elem)
            .dimensions(&[ROWS, WIDTH])
            .vectorize(width)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)
    }
}

/// Where the matrix lies: the alignment its tiles give a line of the rows narrows the width every
/// operand of the producer binds at to one a stored piece of a tile holds.
impl From<&TiledRows> for Placement {
    fn from(rows: &TiledRows) -> Self {
        Self::from(&rows.tensor)
    }
}

/// A packed `[rows, width]` matrix a producer encodes its rows into, under one level of block
/// scales along each row ([`RowStore::Packed`]): its words under their scales, as a product reads
/// its packed lhs.
#[derive(Debug, Clone)]
pub struct PackedRows {
    packed: StoredTensor,
    store: RowStore,
}

impl PackedRows {
    /// `out`, a packed matrix of `rows` rows of `width` elements over the product's lhs dimensions
    /// (`[M, K]`).
    ///
    /// # Errors
    ///
    /// [`PACKED_ROWS_UNSERVED`] for an `out` not over `[M, K]` or not `rows` by `width`, and
    /// [`PACKED_STORE_UNSERVED`](cubek_microkernel::row_store::PACKED_STORE_UNSERVED) for a scheme
    /// the store does not write.
    pub fn new(out: StoredTensor, rows: usize, width: usize) -> Result<Self, Decline> {
        if out.packing().is_none()
            || out.dimensions() != [M, K]
            || [out.matrix_extent(M), out.matrix_extent(K)] != [rows, width]
        {
            return Err(PACKED_ROWS_UNSERVED);
        }
        let store = RowStore::packed(&out.scheme(), width)?;
        Ok(Self { packed: out, store })
    }

    /// The store that encodes the rows into the matrix.
    pub fn store(&self) -> RowStore {
        self.store
    }

    /// The matrix as one tile of its words under their scales under `launcher`, a load as wide as
    /// a line of the producer's rows, `width`, or a word, whichever is wider: a line of the stage
    /// the rows are encoded from lies inside one load.
    ///
    /// # Errors
    ///
    /// As [`StoredTensor::bind`](crate::kernels::tensor::StoredTensor::bind).
    pub fn bind(&self, launcher: &Launcher, width: usize) -> Result<TileTensorLaunch, Decline> {
        self.packed.bind(
            launcher,
            self.packed.operand(),
            self.packed.scheme().num_quants().max(width),
            BlockScales::Fields,
            OuterScale::Kept,
            Delivery::SyncPerUnit,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::select::packed::nvfp4_scheme;
    use cubecl::ir::UIntKind;
    use cubecl::std::tensor::TensorHandle;
    use cubek_test_utils::client;

    /// A packed NVFP4 matrix over `dimensions`, sixteen rows of sixty-four values along `K`.
    fn packed(dimensions: [cubecl::tile::Dimension; 2]) -> StoredTensor {
        let scheme = nvfp4_scheme();
        let u32 = ElemType::UInt(UIntKind::U32);
        let zeros = |shape: [usize; 2]| {
            PlainTensor::of_handle(TensorHandle::zeros(&client(), shape.to_vec(), u32))
        };
        StoredTensor::new(
            zeros([16, 64 / scheme.num_quants()]),
            // A scale byte a block of sixteen values, four a word.
            zeros([16, 1]),
            Some(zeros([1, 1])),
            scheme,
            K,
            None,
            [(dimensions[0], 16), (K, 64)],
        )
    }

    #[test]
    fn a_packed_output_of_other_rows_declines() {
        let declined = PackedRows::new(packed([M, K]), 8, 64).err();

        assert_eq!(declined, Some(PACKED_ROWS_UNSERVED));
    }

    #[test]
    fn a_packed_output_over_other_dimensions_declines() {
        let declined = PackedRows::new(packed([cubek_std::matmul_dimensions::N, K]), 16, 64).err();

        assert_eq!(declined, Some(PACKED_ROWS_UNSERVED));
    }
}
