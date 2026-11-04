use crate::matmul::{MatmulLayout, MatmulOperands, MatmulWorkItem};
use core::num::NonZeroUsize;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::layout::stored::Packing;

/// One unit's step, `columns × depth`, and how the plane's units share `K`.
///
/// The block is counted in reads rather than elements, so one setting loads the same bytes a step
/// at every element size and on every device: a read along the axis the rhs runs is a vector
/// as wide as the device reads its element, and a read across it is one line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleMicrokernelSettings {
    /// The output's columns one unit reads a step: vectors of them for a row rhs, and one
    /// column for a col rhs, which declines more.
    pub columns: NonZeroUsize,
    /// The steps of `K` one unit reads a step, whose loads are issued together rather than left
    /// to the compiler's unrolling: vectors down `K` for a col rhs, single rows of `K` for a
    /// row rhs, which the lhs is read beside in one vector.
    pub depth: NonZeroUsize,
    /// How the plane's units split `K`: how many of them share one run of columns.
    pub k_split: PlaneShuffleKSplit,
    /// Rows one unit holds, a target the product's rows and the plane's width round; every row
    /// where `None`. Fewer rows a unit hold fewer sums in its registers, and leave fewer of the
    /// plane's units to share `K`.
    pub rows_per_unit: Option<NonZeroUsize>,
    /// When a unit loads what a step of `K` reads: as the step contracts, or a step ahead.
    pub fetch: PlaneShuffleFetch,
}

impl PlaneShuffleMicrokernelSettings {
    /// One column four reads deep, read by the whole plane: enough loads in flight a unit to
    /// cover a read's latency.
    pub const DEFAULT: Self = Self {
        columns: NonZeroUsize::MIN,
        depth: NonZeroUsize::new(4).unwrap(),
        k_split: PlaneShuffleKSplit::WholePlane,
        rows_per_unit: None,
        fetch: PlaneShuffleFetch::InStep,
    };

    /// One read deep.
    pub const ONE_READ: Self = Self {
        depth: NonZeroUsize::MIN,
        ..Self::DEFAULT
    };

    /// Two reads deep.
    pub const TWO_READS: Self = Self {
        depth: NonZeroUsize::new(2).unwrap(),
        ..Self::DEFAULT
    };

    /// Two columns a unit, which a col rhs declines.
    pub const TWO_COLUMNS: Self = Self {
        columns: NonZeroUsize::new(2).unwrap(),
        ..Self::DEFAULT
    };

    /// A transaction's worth of units along `K`.
    pub const ONE_TRANSACTION: Self = Self {
        k_split: PlaneShuffleKSplit::OneTransaction,
        ..Self::DEFAULT
    };

    /// One column eight reads deep by units that each own their columns: a row rhs's columns
    /// read at eight rows of `K` with the lhs beside them in one vector.
    pub const EIGHT_READS_ONE_TRANSACTION: Self = Self {
        depth: NonZeroUsize::new(8).unwrap(),
        k_split: PlaneShuffleKSplit::OneTransaction,
        ..Self::DEFAULT
    };

    /// One read deep by a transaction's worth of units: on a col rhs, the most units that
    /// share a column, so the most columns a plane.
    pub const ONE_READ_ONE_TRANSACTION: Self = Self {
        depth: NonZeroUsize::MIN,
        k_split: PlaneShuffleKSplit::OneTransaction,
        ..Self::DEFAULT
    };

    /// Sixteen columns a unit at `f16` where a read is 16 bytes, one row of `K` deep, by the whole
    /// plane: the block a row rhs wins with.
    pub const TWO_ROW_VECTORS: Self = Self {
        columns: NonZeroUsize::new(2).unwrap(),
        depth: NonZeroUsize::MIN,
        ..Self::DEFAULT
    };

    /// Sixteen columns a unit at `f32` where a read is 16 bytes, one row of `K` deep.
    pub const FOUR_ROW_VECTORS: Self = Self {
        columns: NonZeroUsize::new(4).unwrap(),
        depth: NonZeroUsize::MIN,
        ..Self::DEFAULT
    };

    /// The default at two rows a unit: at eight rows, four units side by side along `M`, each
    /// holding a quarter of the sums.
    pub const TWO_ROWS_A_UNIT: Self = Self {
        rows_per_unit: Some(NonZeroUsize::new(2).unwrap()),
        ..Self::DEFAULT
    };

    /// `columns` steps of `N` a unit and one step of `K` a turn, by the whole plane,
    /// `rows_per_unit` rows a unit or every row where `None`: what a packed rhs is read in, its
    /// step the rhs's ([`MatmulStep`](crate::matmul::MatmulStep)).
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn stepping(columns: usize, rows_per_unit: Option<usize>) -> Self {
        Self {
            columns: NonZeroUsize::new(columns).unwrap(),
            depth: NonZeroUsize::MIN,
            k_split: PlaneShuffleKSplit::WholePlane,
            rows_per_unit: match rows_per_unit {
                Some(rows) => Some(NonZeroUsize::new(rows).unwrap()),
                None => None,
            },
            fetch: PlaneShuffleFetch::InStep,
        }
    }

    /// These settings with each step's loads issued as `fetch` says.
    pub const fn fetching(self, fetch: PlaneShuffleFetch) -> Self {
        Self { fetch, ..self }
    }

    /// Units side by side along `M` for `rows` rows on a plane `plane_width` wide: as many as
    /// leave each unit [`rows_per_unit`](Self::rows_per_unit), dividing both the rows and the
    /// plane, and one where a unit holds every row.
    pub fn units_along_m(&self, rows: usize, plane_width: usize) -> usize {
        self.rows_per_unit.map_or(1, |per_unit| {
            (1..=rows / per_unit.get().min(rows))
                .rev()
                .find(|&units| rows.is_multiple_of(units) && plane_width.is_multiple_of(units))
                .unwrap_or(1)
        })
    }
}

impl core::fmt::Display for PlaneShuffleMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "block={}x{} k_split={}",
            self.columns, self.depth, self.k_split
        )?;
        if let Some(rows) = self.rows_per_unit {
            write!(f, " rows_per_unit={rows}")?;
        }
        match self.fetch {
            PlaneShuffleFetch::InStep => Ok(()),
            PlaneShuffleFetch::StepAhead => write!(f, " fetch={}", self.fetch),
        }
    }
}

/// When a plane's units load what a step of `K` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneShuffleFetch {
    /// As the step contracts: each read loaded, decoded and multiplied in turn, the step's loads
    /// left to the compiler to issue ahead of the work they feed.
    InStep,
    /// A step ahead: the next step's reads loaded into registers, undecoded, before this step
    /// decodes and multiplies what it holds, so they are in flight while it does
    /// (`Tile::mma_prefetched`). Against a packed rhs only, read as bound.
    StepAhead,
}

impl PlaneShuffleFetch {
    /// What a step ahead declines: a product whose operands a unit cannot hold undecoded, which is
    /// anything but a packed rhs under byte block scales beside a float lhs.
    pub const STEP_AHEAD_UNHELD: Refusal = Refusal::Declined {
        why: "a step ahead holds a packed rhs under byte block scales beside a float lhs, undecoded",
    };

    /// Checks a product packed as `packings` says can be fetched as this says.
    ///
    /// # Errors
    ///
    /// [`STEP_AHEAD_UNHELD`](Self::STEP_AHEAD_UNHELD), a step ahead of anything but a packed rhs
    /// under byte block scales beside a float lhs.
    pub fn admits(self, packings: &MatmulOperands<Option<Packing>>) -> Result<(), Refusal> {
        let byte_scales = |packing: &Packing| {
            packing
                .scheme
                .block_scale()
                .is_some_and(|scale| ElemType::from_scale_dtype(scale.dtype).size_bits() == 8)
        };
        match (self, &packings.lhs, &packings.rhs) {
            (Self::InStep, _, _) => Ok(()),
            (Self::StepAhead, None, Some(rhs)) if byte_scales(rhs) => Ok(()),
            (Self::StepAhead, _, _) => Err(Self::STEP_AHEAD_UNHELD),
        }
    }
}

impl core::fmt::Display for PlaneShuffleFetch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InStep => "in-step",
            Self::StepAhead => "step-ahead",
        })
    }
}

/// How many of a plane's units share one run of columns, each taking its turn of `K`, named by
/// what it fills rather than counted: the count each names moves with the device and the element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneShuffleKSplit {
    /// The whole plane: one run of columns a plane, its units reading `K` together.
    WholePlane,
    /// The fewest units whose steps fill one memory transaction between them, and the rest of the
    /// plane beside them along `N`. A col rhs's columns run down `K`, so it takes as many units
    /// as one transaction holds a column's step; a row rhs's run along `N`, where the units
    /// beside each other already fill it, so it takes one.
    OneTransaction,
}

impl PlaneShuffleKSplit {
    /// Units along `K` on `hw`, a plane `plane_width` units wide, for a block `depth` reads deep of
    /// `item`'s rhs: a divisor of the plane, since the rest of it lies along `N`.
    pub fn units(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        depth: NonZeroUsize,
        plane_width: usize,
    ) -> usize {
        let column_step_bytes =
            depth.get() * hw.vector_ceiling(item.elements.rhs) * item.elements.rhs.size();
        let wanted = match (self, item.layouts.rhs) {
            (Self::WholePlane, _) => plane_width,
            (Self::OneTransaction, MatmulLayout::Col) => {
                hw.transaction_bytes() / column_step_bytes.max(1)
            }
            (Self::OneTransaction, MatmulLayout::Row | MatmulLayout::Strided) => 1,
        };
        (1..=wanted.clamp(1, plane_width))
            .rev()
            .find(|units| plane_width.is_multiple_of(*units))
            .unwrap_or(1)
    }
}

impl core::fmt::Display for PlaneShuffleKSplit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::WholePlane => "whole-plane",
            Self::OneTransaction => "one-transaction",
        })
    }
}
