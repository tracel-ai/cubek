use cubecl::client::Client;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::{MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::layout::contiguous;
use cubek_std::matmul_axes::{K, M, N};

use crate::kernels::grouped_rows::PaddedRows;
use crate::kernels::m_grouped::{
    MGroupedChunk, MGroupedMatmulKey, MGroupedRowSource, MGroupedSettings,
};
use crate::kernels::routed::stack_extents;
use crate::kernels::stack::{SlabAxis, SlabView};
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::OperandLayout;
use crate::kernels::{
    MATMUL_PACKED_UNREAD, MATMUL_SLABS_TAKE_NO_STORED_TILES, MatmulIndex, MatmulIndexTable,
    MatmulInputs, MatmulRoutine, MatmulRoutineProblem,
};

/// The operands of one grouped product, `out[r] = lhs[r] · rhs[group(r)]`: where the rows come
/// from, sorted by group, the stack of slabs laid `along`, float or packed, and each group's
/// cumulative end row.
///
/// It is a dense product once the groups are padded to whole chunks of rows: each chunk a batch,
/// against the slab of the group it holds, read through the chunks' table ([`MatmulIndex::Slab`]),
/// so a dense routine serves it ([`launch`](Self::launch)). A chunk past the groups' names no slab
/// and sums nothing.
#[derive(Clone)]
pub struct MGroupedMatmulTensors {
    client: Client,
    source: MGroupedRowSource,
    rhs: StoredTensor,
    ends: PlainTensor,
    along: SlabAxis,
}

impl MGroupedMatmulTensors {
    /// The rows `source` holds against `rhs` on `client`'s device, grouped by `ends` (each group's
    /// cumulative end row as a `u32`, one group a slab), the slabs laid along `along`. A float
    /// stack is read where it lies, and copied only where it overlaps itself or is contiguous
    /// along neither axis.
    ///
    /// # Errors
    ///
    /// As [`MatmulInputs::new`] for a float stack, and [`MATMUL_SLABS_TAKE_NO_STORED_TILES`] for
    /// an operand stored in tiles.
    ///
    /// # Panics
    ///
    /// On rows other than a rank-two tensor, a stack other than a rank-two tensor of whole slabs,
    /// a contraction the rows and the stack do not share, or no group.
    pub fn new(
        client: &Client,
        source: impl Into<MGroupedRowSource>,
        rhs: StoredTensor,
        ends: PlainTensor,
        along: SlabAxis,
    ) -> Result<Self, Refusal> {
        let source = source.into();
        let rows = source.buffer();
        let shape = rows.shape();
        assert_eq!(
            shape.num_dims(),
            2,
            "a grouped product's rows are `[rows, k]`: {shape:?}"
        );
        if rows.is_tiled() || rhs.values().is_tiled() {
            return Err(MATMUL_SLABS_TAKE_NO_STORED_TILES);
        }
        let rhs = match rhs {
            StoredTensor::Packed(packed) => StoredTensor::Packed(packed),
            // The rows are padded into a buffer of the launch's own, so only the stack is read
            // where it lies. The slabs are strided views of the stack's buffer, which an
            // overlapping view has no single place for.
            StoredTensor::Plain(rhs) => {
                let (_, rhs) = MatmulInputs::new(client, rows, rhs)?.into_parts();
                StoredTensor::Plain(SlabView::readable(client, rhs.addressable(client)))
            }
        };
        assert_eq!(
            rhs.values().shape().num_dims(),
            2,
            "a grouped product contracts against a `[k, n]` stack"
        );
        let groups = ends.shape().num_elements();
        assert!(groups > 0, "rows are grouped into one slab at least");
        let [k, n] = along.slab_of(stack_extents(&rhs), groups);
        assert_eq!(
            source.k(),
            k,
            "the rows contract against one slab: {} against slabs of [{k}, {n}]",
            source.k()
        );
        Ok(Self {
            client: client.clone(),
            source,
            rhs,
            ends,
            along,
        })
    }

    /// Where the rows come from.
    pub fn source(&self) -> &MGroupedRowSource {
        &self.source
    }

    /// The same product over the rows `source` holds, grouped by `ends`: what a candidate is
    /// measured on when the call's own layout cannot be trusted.
    pub(crate) fn with_rows(&self, source: MGroupedRowSource, ends: PlainTensor) -> Self {
        Self {
            source,
            ends,
            ..self.clone()
        }
    }

    /// Each group's cumulative end row, `u32`.
    pub(crate) fn ends(&self) -> &PlainTensor {
        &self.ends
    }

    /// The client the operands live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Slabs in the stack, one per group.
    pub fn groups(&self) -> usize {
        self.ends.shape().num_elements()
    }

    /// One slab's `[k, n]`.
    pub fn slab_shape(&self) -> [usize; 2] {
        self.along.slab_of(stack_extents(&self.rhs), self.groups())
    }

    /// The key the grouped autotune measures these operands under.
    pub fn key(&self) -> MGroupedMatmulKey {
        let [k, n] = self.slab_shape();
        MGroupedMatmulKey::new(
            self.source.buffer().elem,
            k,
            n,
            self.groups(),
            self.source.rows(),
            self.rhs.kind(),
            OperandLayout::along(contiguous(&self.rhs.values().meta(), [K, N])),
            self.along,
            self.source.kind(),
        )
    }

    /// Whether `routine` at `settings` prepares, on `hw`, one chunk's product against one slab,
    /// `rows` rows of it: what a launch padding the groups to such chunks runs on each.
    ///
    /// # Errors
    ///
    /// As [`launch`](Self::launch).
    pub fn served_by<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &MGroupedSettings<R::Settings>,
    ) -> Result<(), Refusal>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let rows = self.chunk_rows(hw, routine, settings)?;
        routine
            .prepare(hw, &self.slab_problem(rows)?, &settings.routine)
            .map(|_| ())
    }

    /// The product on `routine` at `settings`, prepared for `hw`, its groups padded to the chunks
    /// the settings name: the output handed back padded as the rows were, for a next step that
    /// reads it padded.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the product at these settings, or why the routine declines a table or
    /// the stack as it is stored.
    ///
    /// # Panics
    ///
    /// On no row: a layout holds one at least.
    pub fn launch<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &MGroupedSettings<R::Settings>,
    ) -> Result<PaddedRows, Refusal>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        assert!(
            self.source.rows() > 0,
            "a padded product has a row at least"
        );
        let chunk_rows = self.chunk_rows(hw, routine, settings)?;
        let stack = self.stack()?;
        let (lhs, layout) = self
            .source
            .clone()
            .pad(&self.client, self.ends.clone(), chunk_rows)
            .into_parts();
        let [k, n] = self.slab_shape();
        let chunks = lhs.shape()[0] / chunk_rows;
        // Each chunk is a batch of its rows, every one against the slab of the group it holds.
        let rows = PlainTensor::of_handle(lhs.clone())
            .as_rows(chunks * chunk_rows, k)
            .viewed(Metadata::new(
                [chunks, chunk_rows, k],
                [chunk_rows * k, k, 1],
            ));
        let table = MatmulIndexTable {
            index: MatmulIndex::Slab {
                slabs: self.groups(),
            },
            entries: PlainTensor::of_handle(layout.table()),
        };
        let (out, bound) = PlainTensor::empty_contiguous(
            &self.client,
            Shape::new([chunks * chunk_rows, n]),
            lhs.dtype,
        );
        let bound = bound.viewed(Metadata::new(
            [chunks, chunk_rows, n],
            [chunk_rows * n, n, 1],
        ));
        routine.launch_indexed_into(
            hw,
            &self.client,
            rows,
            stack,
            table,
            bound,
            &settings.routine,
        )?;
        Ok(PaddedRows::new(out, layout))
    }

    /// The rows each group is padded to a whole number of, as `settings` name them, on `hw`.
    ///
    /// # Errors
    ///
    /// For the rows of a cube's tile, [`MATMUL_PACKED_UNREAD`] for a packed stack, and why `hw`
    /// cannot run the product at these settings.
    fn chunk_rows<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &MGroupedSettings<R::Settings>,
    ) -> Result<usize, Refusal>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        match settings.chunk {
            MGroupedChunk::Rows(rows) => Ok(rows.get()),
            MGroupedChunk::CubeTile => self.rows_a_cube_holds(hw, routine, &settings.routine),
        }
    }

    /// The rows one cube of `routine` at `settings` holds against a float stack, prepared for
    /// `hw` over every grouped row as one dense product.
    ///
    /// # Errors
    ///
    /// [`MATMUL_PACKED_UNREAD`] for a packed stack, whose blocks the rows are not counted in, and
    /// why `hw` cannot run the product at these settings.
    fn rows_a_cube_holds<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &R::Settings,
    ) -> Result<usize, Refusal>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        if matches!(self.rhs, StoredTensor::Packed(_)) {
            return Err(MATMUL_PACKED_UNREAD);
        }
        let problem = self.slab_problem(self.source.rows())?;
        let (blueprint, _) = routine.prepare(hw, &problem, settings)?;
        Ok(blueprint.partitioning.cube_box().extent(M))
    }

    /// `rows` rows against one slab, as a dense product states it, the slab aligned as a slab of
    /// the stack is, so a line never reads across two: float, or packed as the stack is.
    ///
    /// # Errors
    ///
    /// What reading a packed stack slab by slab declines.
    fn slab_problem(&self, rows: usize) -> Result<MatmulRoutineProblem, Refusal> {
        let elem = self.source.buffer().elem;
        match &self.rhs {
            StoredTensor::Packed(packed) => {
                let stack = packed.clone().stacked(self.groups(), self.along.axis())?;
                let [k, n] = self.slab_shape();
                let packings = MatmulOperands {
                    rhs: Some(stack.packing()),
                    ..MatmulOperands::all(None)
                };
                Ok(MatmulRoutineProblem::packed(rows, n, k, elem, packings))
            }
            StoredTensor::Plain(rhs) => {
                let slabs = SlabView::new(rhs, self.groups(), self.along);
                let rhs_layout = MatmulLayout::new(
                    &Metadata::new(slabs.shape, slabs.strides),
                    MatmulOperand::Rhs,
                );
                Ok(MatmulRoutineProblem::slab(
                    rows,
                    elem,
                    rhs,
                    (slabs.shape, slabs.strides),
                    rhs_layout,
                ))
            }
        }
    }

    /// The stack as the product reads it: a float one as its slabs, `[groups, k, n]` over the
    /// buffer it lies in, a packed one its words and scales read slab by slab.
    fn stack(&self) -> Result<StoredTensor, Refusal> {
        match &self.rhs {
            StoredTensor::Packed(packed) => Ok(StoredTensor::Packed(
                packed.clone().stacked(self.groups(), self.along.axis())?,
            )),
            StoredTensor::Plain(rhs) => {
                let slabs = SlabView::new(rhs, self.groups(), self.along);
                Ok(StoredTensor::Plain(slabs.slabs_of(rhs)))
            }
        }
    }
}
