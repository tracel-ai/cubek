use crate::kernels::tensor::{PackedTensor, Placement, PlainTensor};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{Bound, Delivery, Refusal};
use cubecl::tile::{Launcher, TileTensorLaunch};
use cubek_microkernel::row_store::{ROWS, RowStore, WIDTH};
use cubek_std::layout::stored::BlockScales;
use cubek_std::matmul_axes::{K, M};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// What a tiled launch declines before preparing anything: an output that is not a float matrix
/// of the producer's rows stored in tiles, at the producer's element.
pub const TILED_ROWS_UNSERVED: Refusal = Refusal::Declined {
    why: "a tiled output is a float matrix of the producer's rows, stored in tiles, at its element",
};

/// A matrix of a producer's rows, allocated by the caller, laid out the way a product reads its
/// lhs: packed under block scales along each row ([`PackedRows`]), or float in the product's
/// storage tiles ([`TiledRows`]). It is one type because a family's tuned launch writes either one
/// into its output the same way: the routine its autotune elected, or its floor.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, moved into it once and read there"
)]
pub enum LhsOutput {
    /// A packed `[rows, width]` matrix under one level of block scales along each row.
    Packed(PackedTensor),
    /// A float `[rows, width]` matrix stored in tiles.
    Tiled(PlainTensor),
}

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
    ) -> Result<Self, Refusal> {
        let fits = out.is_tiled() && out.elem == element && out.shape().as_slice() == [rows, width];
        match fits {
            true => Ok(Self { tensor: out }),
            false => Err(TILED_ROWS_UNSERVED),
        }
    }

    /// The matrix bound under `launcher` over the rows and the row span, `width` elements a line.
    pub fn bind(&self, launcher: &Launcher, width: usize) -> Bound {
        launcher
            .arg(self.tensor.binding.clone(), self.tensor.elem)
            .axes(&[ROWS, WIDTH])
            .vectorize(width)
            .build()
            .expect(ADMITTED_PLAN_RUNS)
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
    packed: PackedTensor,
    store: RowStore,
}

impl PackedRows {
    /// `out`, a packed matrix of `rows` rows of `width` elements over the product's lhs axes
    /// (`[M, K]`).
    ///
    /// # Errors
    ///
    /// [`PACKED_STORE_UNSERVED`](cubek_microkernel::row_store::PACKED_STORE_UNSERVED) for a scheme
    /// the store does not write.
    ///
    /// # Panics
    ///
    /// On an `out` not over `[M, K]` or not `rows` by `width`.
    pub fn new(out: PackedTensor, rows: usize, width: usize) -> Result<Self, Refusal> {
        assert_eq!(out.axes(), [M, K], "a packed output is a product's lhs");
        assert_eq!(
            [out.extent(M), out.extent(K)],
            [rows, width],
            "a packed output holds the producer's rows"
        );
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
    pub fn bind(&self, launcher: &Launcher, width: usize) -> TileTensorLaunch {
        self.packed.bind(
            launcher,
            self.packed.scheme().num_quants().max(width),
            BlockScales::Fields,
            Delivery::SyncPerUnit,
        )
    }
}
