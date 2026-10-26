use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;

use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::{
    MATMUL_SLABS_TAKE_NO_STORED_TILES, MatmulIndex, MatmulIndexTable, MatmulInputs, MatmulRoutine,
};

/// The operands of one K-grouped product, `out[g] = lhs[rows of g]ᵀ · rhs[rows of g]`: the two row
/// sets and each group's cumulative end row.
///
/// It is a dense product: each group a batch, contracting both operands over its run of rows, the
/// ends the table both read ([`MatmulIndex::RunOfK`]), and each batch's output one slab of the
/// stack the groups' slabs lie side by side in, so a dense routine serves it
/// ([`launch`](Self::launch)).
#[derive(Clone)]
pub struct KGroupedMatmulTensors {
    client: Client,
    lhs: PlainTensor,
    rhs: PlainTensor,
    ends: PlainTensor,
}

impl KGroupedMatmulTensors {
    /// `lhs[rows, m]` and `rhs[rows, n]` on `client`'s device, their rows grouped by `ends`, each
    /// group's cumulative end row as a `u32`, one group a slab.
    ///
    /// The ends are the caller's to get right, since checking them would read them back:
    /// non-decreasing, the last one `rows`. Wrong ones cost the answer and nothing else: a group
    /// that runs backwards reads no row, and no row past `rows` is read.
    ///
    /// # Errors
    ///
    /// As [`MatmulInputs::new`], and [`MATMUL_SLABS_TAKE_NO_STORED_TILES`] for an operand stored
    /// in tiles.
    ///
    /// # Panics
    ///
    /// On operands other than two rank-two tensors of one row count, or ends other than one `u32`
    /// per group, at least one.
    pub fn new(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        ends: PlainTensor,
    ) -> Result<Self, Refusal> {
        if lhs.is_tiled() || rhs.is_tiled() {
            return Err(MATMUL_SLABS_TAKE_NO_STORED_TILES);
        }
        let (lhs, rhs) = MatmulInputs::new(client, lhs, rhs)?.into_parts();
        let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
        assert!(
            lhs_shape.num_dims() == 2 && rhs_shape.num_dims() == 2,
            "a K-grouped product is `[rows, m]` against `[rows, n]`: {lhs_shape:?} and \
             {rhs_shape:?}"
        );
        assert_eq!(
            rhs_shape[0], lhs_shape[0],
            "both operands hold the grouped rows"
        );
        assert_eq!(
            ends.elem,
            ElemType::UInt(UIntKind::U32),
            "the groups' ends are `u32` rows"
        );
        assert_eq!(ends.shape().rank(), 1, "the groups' ends are one per group");
        assert!(
            ends.shape().num_elements() > 0,
            "rows are grouped into one slab at least"
        );
        Ok(Self {
            client: client.clone(),
            lhs,
            rhs,
            ends,
        })
    }

    /// The client the operands live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The output's shape, `[m, groups · n]`: group `g` owns columns `g · n..(g + 1) · n`, as the
    /// stack an M-grouped product reads.
    pub fn output_shape(&self) -> [usize; 2] {
        let groups = self.ends.shape().num_elements();
        [self.lhs.shape()[1], groups * self.rhs.shape()[1]]
    }

    /// The product on `routine` at `settings`, prepared for `hw`. No row is a zero stack, with no
    /// launch.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the product at these settings, or why the routine declines a table.
    pub fn launch<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Refusal>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let shape = self.output_shape();
        let elem = self.lhs.elem;
        if self.lhs.shape()[0] == 0 {
            return Ok(TensorHandle::zeros(&self.client, shape, elem));
        }
        let [m, _] = shape;
        let (groups, n) = (self.ends.shape().num_elements(), self.rhs.shape()[1]);
        // The lhs read transposed, `[m, rows]`, so the grouped rows are its contraction.
        let lhs = self.lhs.transposed();
        let table = MatmulIndexTable {
            index: MatmulIndex::RunOfK,
            entries: self.ends.clone(),
        };
        let (out, bound) = PlainTensor::empty_contiguous(&self.client, Shape::from(shape), elem);
        // Batch `g` of the `[groups, m, n]` view is group `g`'s slab of the stack.
        let bound = bound.viewed(Metadata::new([groups, m, n], [n, groups * n, 1]));
        routine.launch_indexed_into(
            hw,
            &self.client,
            lhs,
            StoredTensor::Plain(self.rhs.clone()),
            table,
            bound,
            settings,
        )?;
        Ok(out)
    }
}
