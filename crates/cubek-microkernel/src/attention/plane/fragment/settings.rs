use crate::attention::ValueLines;
use crate::matmul::MatmulWorkItem;
use crate::matmul::plane::{PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings};
use core::num::NonZeroUsize;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{FragmentSource, ProductStep, Scratch};
use cubek_std::Hardware;

/// The shape of the tile a plane holds along the queries and along a stage of keys, in units of the
/// device's matrix unit for the operands
/// ([`matrix_unit`](crate::matmul::plane::PlaneFragmentMicrokernel::matrix_unit)), how much of its
/// output a correction bounces through shared memory at once, and where it keeps the score. The
/// instruction is not a setting: the tile layer picks it where the score is held, and it cuts the
/// same tile whichever it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionMicrokernelSettings {
    pub seq_q: NonZeroUsize,
    pub seq_kv: NonZeroUsize,
    /// One fragment at a time, three `sync_plane` each, or the whole grid at once, three a stage
    /// for a window as large as the plane's output.
    pub scratch: Scratch,
    /// Where the plane keeps the score between its two multiplications.
    pub score: ScoreHolder,
    /// Where the plane's query stays across the keys' walk.
    pub query: QueryHolder,
    /// The dimension the stage's values lie contiguous along: the value dim, as their buffer holds
    /// them, which a stage fills whole lines of, or the keys, their contraction, which a fragment
    /// loads whole runs of. A stage several planes share is filled once and loaded by each.
    pub value_lines: ValueLines,
}

/// Where a plane's query stays across its walk over the keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryHolder {
    /// The stage in shared memory its cube or the plane fills before the walk, which each step
    /// loads its fragments from.
    Stage,
    /// The plane's registers: its fragments loaded once from global memory. The stage's bytes hold
    /// keys instead, so more planes share each stage of them, and no step reads the query from
    /// shared memory. It is the `A` of the score's manual `mma`, so it asks for a score kept in
    /// fragments ([`ScoreHolder::Fragments`]) and a query read at the element it is stored in.
    Registers,
}

/// Where a plane keeps a stage's score between its two multiplications. The tile layer picks the
/// instruction from it: a score opened in windows takes the accumulator's pick, one kept in its
/// fragments the pick for a sum contracted again where it lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreHolder {
    /// The plane's windows of shared memory: the score drained into one, the softmax read there,
    /// the probabilities cast into the other and loaded again as fragments. What a cmma fragment,
    /// whose cells lie where the instruction keeps them, is read through.
    Window,
    /// The score's own manual-mma fragments: the softmax reads them where the plane's units hold
    /// them, and the value contracts them, cast, in the same registers. Each unit keeps the state
    /// of the rows it holds cells of, so the plane's state merges with no other holder's. Only a
    /// device whose instruction lays its `A` as it lays its accumulator keeps it so.
    Fragments,
}

impl ScoreHolder {
    /// The plane holding the sum kept here, `item` over the plane's `tile`, its settings `grid`:
    /// drained into a window, the sum of its operands' values; held in registers, a sum kept for
    /// the next product ([`PlaneFragmentMicrokernel::keeping`]), which the kernel opens against
    /// its lhs.
    pub(crate) fn sum_plane(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        grid: PlaneFragmentMicrokernelSettings,
        tile: [usize; 3],
    ) -> Option<PlaneFragmentMicrokernel> {
        let step = Self::sum_step(item, tile);
        match self {
            ScoreHolder::Window => PlaneFragmentMicrokernel::draining(hw, item, grid, step),
            ScoreHolder::Fragments => {
                let kept = ProductStep {
                    rhs: step.lhs,
                    ..step
                };
                PlaneFragmentMicrokernel::keeping(hw, item, grid, kept)
            }
        }
    }

    /// [`sum_plane`](Self::sum_plane) for a plane holding the sum's lhs in registers across its
    /// products ([`QueryHolder::Registers`]), which only a sum held in registers contracts.
    pub(crate) fn held_lhs_sum_plane(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        grid: PlaneFragmentMicrokernelSettings,
        tile: [usize; 3],
    ) -> Option<PlaneFragmentMicrokernel> {
        let step = Self::sum_step(item, tile);
        match self {
            ScoreHolder::Window => None,
            ScoreHolder::Fragments => {
                PlaneFragmentMicrokernel::keeping_held_lhs(hw, item, grid, step)
            }
        }
    }

    /// The step of `item`'s sum over the plane's `tile`, its operands read as values.
    fn sum_step(item: &MatmulWorkItem, tile: [usize; 3]) -> ProductStep {
        ProductStep {
            shape: tile,
            holder: ComputeScope::Plane,
            lhs: FragmentSource::Values(item.elements.lhs),
            rhs: FragmentSource::Values(item.elements.rhs),
            sum: item.sum,
        }
    }

    /// The plane of a product that reads the sum `kept` holds here as its `A`, `item` over the
    /// plane's `tile`, against `rhs`, its `B` as the kernel opens the product's sum on it, its
    /// settings `grid`: out of the window, cast to the operands' element; held in registers, runs of
    /// `kept`'s cells as they lie ([`PlaneFragmentMicrokernel::kept_sum`]). Held cells name no
    /// element, so the tile picks nothing for them against values a stage's fill decodes
    /// ([`FragmentSource::Decoded`]).
    pub(crate) fn reader_plane(
        self,
        hw: &Hardware,
        kept: &PlaneFragmentMicrokernel,
        item: &MatmulWorkItem,
        rhs: FragmentSource,
        grid: PlaneFragmentMicrokernelSettings,
        tile: [usize; 3],
    ) -> Option<PlaneFragmentMicrokernel> {
        let step = self.reader_step(kept, item, rhs, tile);
        PlaneFragmentMicrokernel::draining(hw, item, grid, step)
    }

    /// [`reader_plane`](Self::reader_plane) for a product whose sum's rows are scaled between its
    /// stages, as the forward's output is by its online softmax
    /// ([`PlaneFragmentMicrokernel::rescaling`]).
    pub(crate) fn rescaled_reader_plane(
        self,
        hw: &Hardware,
        kept: &PlaneFragmentMicrokernel,
        item: &MatmulWorkItem,
        rhs: FragmentSource,
        grid: PlaneFragmentMicrokernelSettings,
        tile: [usize; 3],
    ) -> Option<PlaneFragmentMicrokernel> {
        let step = self.reader_step(kept, item, rhs, tile);
        PlaneFragmentMicrokernel::rescaling(hw, item, grid, step)
    }

    /// The step of a product reading the sum `kept` holds here as its `A`, against `rhs`.
    fn reader_step(
        self,
        kept: &PlaneFragmentMicrokernel,
        item: &MatmulWorkItem,
        rhs: FragmentSource,
        tile: [usize; 3],
    ) -> ProductStep {
        let lhs = match self {
            ScoreHolder::Window => FragmentSource::Values(item.elements.lhs),
            ScoreHolder::Fragments => kept.kept_sum(),
        };
        ProductStep {
            shape: tile,
            holder: ComputeScope::Plane,
            lhs,
            rhs,
            sum: item.sum,
        }
    }
}

impl PlaneFragmentAttentionMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        seq_q: NonZeroUsize::new(2).unwrap(),
        seq_kv: NonZeroUsize::new(2).unwrap(),
        scratch: Scratch::OneTile,
        score: ScoreHolder::Fragments,
        query: QueryHolder::Stage,
        value_lines: ValueLines::ValueDim,
    };

    /// The default with every fragment of the grid bounced at once.
    pub const WHOLE_GRID: Self = Self {
        scratch: Scratch::WholeGrid,
        ..Self::DEFAULT
    };

    /// The default reading the score through its windows.
    pub const THROUGH_WINDOWS: Self = Self::DEFAULT.held_in(ScoreHolder::Window);

    /// The default with `rows` units of rows and `keys` of keys.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn fragments(rows: usize, keys: usize) -> Self {
        Self {
            seq_q: NonZeroUsize::new(rows).unwrap(),
            seq_kv: NonZeroUsize::new(keys).unwrap(),
            ..Self::DEFAULT
        }
    }

    /// These settings keeping the query in `query`.
    pub const fn querying(self, query: QueryHolder) -> Self {
        Self { query, ..self }
    }

    /// These settings holding the values along `value_lines`.
    pub const fn along(self, value_lines: ValueLines) -> Self {
        Self {
            value_lines,
            ..self
        }
    }

    /// These settings keeping the score in `score`.
    pub const fn held_in(self, score: ScoreHolder) -> Self {
        Self { score, ..self }
    }
}

impl core::fmt::Display for PlaneFragmentAttentionMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fragments={}x{}", self.seq_q, self.seq_kv)?;
        match self.scratch {
            Scratch::WholeGrid => write!(f, " scratch=whole")?,
            Scratch::OneTile | Scratch::None => {}
        }
        match self.score {
            ScoreHolder::Fragments => {}
            ScoreHolder::Window => write!(f, " score=window")?,
        }
        match self.query {
            QueryHolder::Stage => Ok(()),
            QueryHolder::Registers => write!(f, " query=registers"),
        }?;
        match self.value_lines {
            ValueLines::ValueDim => Ok(()),
            ValueLines::Keys => write!(f, " values=along_keys"),
        }
    }
}
