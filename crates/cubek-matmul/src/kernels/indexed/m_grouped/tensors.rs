#[cfg(doc)]
use crate::kernels::indexed::m_grouped::{
    GATED_ROWS_UNREAD, GROUPED_ROWS_UNREAD, ROUTE_SORT_UNREAD,
};
use crate::kernels::indexed::m_grouped::{
    GroupedRows, MGroupedChunk, MGroupedFloors, MGroupedMatmulKey, MGroupedRowSource,
    MGroupedSettings, PaddedRows, RouteSort, upload,
};
#[cfg(doc)]
use crate::kernels::indexed::stack::{GROUP_ENDS_UNREAD, MATMUL_STACK_NOT_WHOLE_SLABS};
use crate::kernels::indexed::stack::{SlabDimension, SlabView, admit_ends};
use crate::kernels::select::problem::OperandKind;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::{
    MATMUL_SLABS_TAKE_NO_STORED_TILES, MatmulIndex, MatmulIndexTable, MatmulInputs, MatmulRoutine,
    MatmulRoutineProblem,
};
use crate::kernels::{MatmulTuneKey, OperandLayout};
use cubecl::client::Client;
use cubecl::tile::launch::Decline;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::{MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::layout::contiguous;
use cubek_std::matmul_dimensions::{K, N};
use cubek_std::routine::Breakable;

/// A grouped product whose rows' contraction is not its slabs'.
pub const M_GROUPED_SHAPES_DISAGREE: Decline = Decline::Unserved {
    why: "a grouped product is [rows, k] against slabs of [k, n]",
};

/// What a launch pads its rows on: the rows and the ends [`MGroupedMatmulTensors::new`] admitted,
/// in chunks of a row at least.
const PADDED_AS_ADMITTED: &str =
    "the rows and the ends should be as the product admitted them, in chunks of a row at least";

/// A grouped product padded over no row: a layout holds one at least.
pub const M_GROUPED_NO_ROW: Decline = Decline::Unserved {
    why: "a padded product has a row at least",
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
    along: SlabDimension,
}

impl MGroupedMatmulTensors {
    /// The rows `source` holds against `rhs` on `client`'s device, grouped by `ends` (each group's
    /// cumulative end row as a `u32`, one group a slab), the slabs laid along `along`. A float
    /// stack is read where it lies, and copied only where it overlaps itself or is contiguous
    /// along neither dimension.
    ///
    /// # Assumptions
    ///
    /// As [`MGroupedRowSource::pad`].
    ///
    /// # Guarantees
    ///
    /// As [`MGroupedRowSource::pad`].
    ///
    /// # Errors
    ///
    /// As [`MatmulInputs::new`] for a float stack, [`MATMUL_SLABS_TAKE_NO_STORED_TILES`] for an
    /// operand stored in tiles, [`GROUPED_ROWS_UNREAD`], [`ROUTE_SORT_UNREAD`] or
    /// [`GATED_ROWS_UNREAD`] for the rows, [`MATMUL_STACK_NOT_WHOLE_SLABS`],
    /// [`GROUP_ENDS_UNREAD`], and [`M_GROUPED_SHAPES_DISAGREE`].
    pub fn new(
        client: &Client,
        source: impl Into<MGroupedRowSource>,
        rhs: StoredTensor,
        ends: PlainTensor,
        along: SlabDimension,
    ) -> Result<Self, Decline> {
        let source = source.into();
        admit_ends(&ends)?;
        let groups = ends.shape().num_elements();
        source.admit(groups)?;
        along.admit(&rhs, groups)?;
        let [k, _] = along.slab_shape(&rhs, groups);
        if source.k() != k {
            return Err(M_GROUPED_SHAPES_DISAGREE);
        }
        let rows = source.buffer();
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

    /// The same product and stack, the call's rows spread evenly over its groups: operands whose
    /// layout is the one they state whatever the call's buffers hold, for a caller that measures
    /// the product on operands of its own. A routed source's places name its own routes, and a
    /// gated source's layout is built on the spread ends.
    ///
    /// The sort that makes the call's `ends` and route places can be discarded, as a process mode
    /// that discards launches does, leaving whatever the buffers held, which a kernel reading
    /// them as a layout runs on as garbage. These are uploaded rather than launched, so such a
    /// mode keeps them.
    pub fn with_spread_rows(&self) -> Self {
        let client = &self.client;
        let rows = self.source.rows();
        let groups = self.groups();
        let spread: Vec<u32> = (1..=groups)
            .map(|group| (group * rows / groups) as u32)
            .collect();
        let ends = PlainTensor::of_handle(upload(client, &spread));
        let source = match &self.source {
            MGroupedRowSource::Sorted(rows) => MGroupedRowSource::Sorted(rows.clone()),
            MGroupedRowSource::Routed { tokens, sort } => {
                let places: Vec<u32> = (0..rows as u32).collect();
                MGroupedRowSource::Routed {
                    tokens: tokens.clone(),
                    sort: RouteSort {
                        route_at_place: PlainTensor::of_handle(upload(client, &places)),
                        slots: sort.slots,
                    },
                }
            }
            MGroupedRowSource::Gated(both) => {
                let align = both.layout().align();
                let layout = GroupedRows::uploaded(client, &spread, rows, align);
                MGroupedRowSource::Gated(both.placed_by(layout))
            }
        };
        Self {
            client: client.clone(),
            source,
            rhs: self.rhs.clone(),
            ends,
            along: self.along,
        }
    }

    /// Whether some floor listed for the stack's kind prepares the product on `hw`
    /// ([`MGroupedMatmulCandidates::floors`](super::MGroupedMatmulCandidates::floors)): where
    /// none does, no candidate is known to, and a caller names what runs instead.
    ///
    /// # Errors
    ///
    /// [`M_GROUPED_NO_ROW`], or the last floor's decline, where none serves.
    pub fn served_on_floor(&self, hw: &Hardware) -> Result<(), Decline> {
        if self.source.rows() == 0 {
            return Err(M_GROUPED_NO_ROW);
        }
        MGroupedFloors::LISTED.served(hw, self)
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
        self.along.slab_shape(&self.rhs, self.groups())
    }

    /// Whether the stack is float or packed, and how.
    pub fn stack_kind(&self) -> OperandKind {
        self.rhs.kind()
    }

    /// The key the grouped autotune measures these operands under.
    pub fn key(&self) -> MGroupedMatmulKey {
        let [k, n] = self.slab_shape();
        MGroupedMatmulKey::new(
            self.source.buffer().elem,
            k,
            n,
            self.groups(),
            MatmulTuneKey::anchored(self.source.rows(), Self::ROWS_KEYED),
            self.stack_kind(),
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
        settings: &MGroupedSettings<R::Settings>,
    ) -> Result<(), Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let rows = self.chunk_rows::<Mk, SM, R>(hw, settings)?;
        R::prepare(hw, &self.slab_problem(rows)?, &settings.routine).map(|_| ())
    }

    /// The product on `routine` at `settings`, prepared for `hw`, its groups padded to the chunks
    /// the settings name: the output handed back padded as the rows were, for a next step that
    /// reads it padded.
    ///
    /// # Errors
    ///
    /// [`M_GROUPED_NO_ROW`], why `hw` cannot run the product at these settings, or why the routine
    /// declines a table or the stack as it is stored.
    pub fn launch<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &MGroupedSettings<R::Settings>,
    ) -> Result<PaddedRows, Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        if self.source.rows() == 0 {
            return Err(M_GROUPED_NO_ROW);
        }
        let chunk_rows = self.chunk_rows::<Mk, SM, R>(hw, settings)?;
        let stack = self.along.read_slabs(&self.rhs, self.groups())?;
        let (lhs, layout) = self
            .source
            .clone()
            .pad(&self.client, self.ends.clone(), chunk_rows)
            .or_broken(PADDED_AS_ADMITTED)?
            .into_parts();
        let [k, n] = self.slab_shape();
        let (chunks, elem) = (lhs.shape()[0] / chunk_rows, lhs.dtype);
        // Each chunk is a batch of its rows, every one against the slab of the group it holds.
        let rows = PlainTensor::of_handle(lhs)
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
        let (out, bound) =
            PlainTensor::empty_contiguous(&self.client, Shape::new([chunks * chunk_rows, n]), elem);
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
        Ok(PaddedRows::laid_out(out, layout))
    }
}

impl MGroupedMatmulTensors {
    /// The most rows a key tells apart: past it a grid is full whatever the count.
    const ROWS_KEYED: usize = 65536;
}

impl MGroupedMatmulTensors {
    /// Each group's cumulative end row, `u32`.
    #[cfg(test)]
    pub(crate) fn ends(&self) -> &PlainTensor {
        &self.ends
    }

    /// The rows each group is padded to a whole number of, as `settings` name them, on `hw`.
    ///
    /// # Errors
    ///
    /// What [`rows_a_cube_holds`](Self::rows_a_cube_holds) declines, for the rows of a cube's tile.
    fn chunk_rows<Mk, SM, R>(
        &self,
        hw: &Hardware,
        settings: &MGroupedSettings<R::Settings>,
    ) -> Result<usize, Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        match settings.chunk {
            MGroupedChunk::Rows(rows) => Ok(rows.get()),
            MGroupedChunk::CubeTile => self.rows_a_cube_holds::<Mk, SM, R>(hw, &settings.routine),
        }
    }

    /// The rows one cube of `R` at `settings` holds against the stack, float or packed,
    /// prepared for `hw` over every grouped row as one dense product.
    ///
    /// # Errors
    ///
    /// What reading a packed stack slab by slab declines, and why `hw` cannot run the product at
    /// these settings.
    fn rows_a_cube_holds<Mk, SM, R>(
        &self,
        hw: &Hardware,
        settings: &R::Settings,
    ) -> Result<usize, Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let problem = self.slab_problem(self.source.rows())?;
        let (blueprint, _) = R::prepare(hw, &problem, settings)?;
        Ok(blueprint.partitioning.rows() as usize)
    }

    /// `rows` rows against one slab, as a dense product states it, the slab aligned as a slab of
    /// the stack is, so a line never reads across two: float, or packed as the stack is.
    ///
    /// # Errors
    ///
    /// What reading a packed stack slab by slab declines.
    fn slab_problem(&self, rows: usize) -> Result<MatmulRoutineProblem, Decline> {
        let elem = self.source.buffer().elem;
        match &self.rhs {
            StoredTensor::Packed(packed) => {
                let stack = packed
                    .clone()
                    .stacked(self.groups(), self.along.dimension())?;
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
}
