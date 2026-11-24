use crate::matmul::{MatmulDimensions, MatmulLayout, MatmulWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::Space;
use cubecl::tile::layout::{Run, VectorTile};
use cubek_std::Hardware;

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
    /// Whether a packed rhs is read a scale block at a time
    /// ([`Operand::by_block`](cubek_std::layout::stored::Operand::by_block)): the step along `K`
    /// is one block rather than a word of scales, so a `K` that is no whole number of words
    /// closes. Declined against an rhs that reads none so.
    pub by_block: bool,
}

impl PlaneShuffleMicrokernelSettings {
    /// One column four reads deep, read by the whole plane: enough loads in flight a unit to
    /// cover a read's latency.
    pub const DEFAULT: Self = Self {
        columns: NonZeroUsize::MIN,
        depth: NonZeroUsize::new(4).unwrap(),
        k_split: PlaneShuffleKSplit::WholePlane,
        rows_per_unit: None,
        by_block: false,
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
        k_split: PlaneShuffleKSplit::Run(Run::Transaction),
        ..Self::DEFAULT
    };

    /// One column eight reads deep by units that each own their columns: a row rhs's columns
    /// read at eight rows of `K` with the lhs beside them in one vector.
    pub const EIGHT_READS_ONE_TRANSACTION: Self = Self {
        depth: NonZeroUsize::new(8).unwrap(),
        k_split: PlaneShuffleKSplit::Run(Run::Transaction),
        ..Self::DEFAULT
    };

    /// One read deep by a transaction's worth of units: on a col rhs, the most units that
    /// share a column, so the most columns a plane.
    pub const ONE_READ_ONE_TRANSACTION: Self = Self {
        depth: NonZeroUsize::MIN,
        k_split: PlaneShuffleKSplit::Run(Run::Transaction),
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
    /// step the rhs's ([`MicroRegion`](cubecl::tile::MicroRegion)).
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
            by_block: false,
        }
    }

    /// These settings reading a packed rhs a scale block at a time
    /// ([`by_block`](Self::by_block)).
    pub const fn by_block(self) -> Self {
        Self {
            by_block: true,
            ..self
        }
    }

    /// These settings with as many units along `K` as read one memory transaction whole, the rest
    /// of the plane along `N` ([`Run::Transaction`]).
    pub const fn one_transaction(self) -> Self {
        Self {
            k_split: PlaneShuffleKSplit::Run(Run::Transaction),
            ..self
        }
    }

    /// These settings with as many units along `K` as read one stored tile of a packed rhs whole,
    /// the rest of the plane along `N` ([`Run::StoredTile`]).
    pub const fn one_tile(self) -> Self {
        Self {
            k_split: PlaneShuffleKSplit::Run(Run::StoredTile),
            ..self
        }
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
        match self.by_block {
            true => write!(f, " by_block"),
            false => Ok(()),
        }
    }
}

/// How many of a plane's units share one run of columns, each taking its turn of `K`: the whole
/// plane, or as many as read one [`Run`] of the rhs whole, the rest of the plane beside them along
/// `N`. The run is named rather than counted: how many reads it holds follows from how the rhs is
/// stored ([`VectorTile::run_loads`]), and moves with the device and the element.
///
/// A transaction runs where the rhs lays its bytes together: down `K` for a col rhs, across the
/// rows for a row rhs, whose units beside each other already fill it, so it takes one unit; in
/// tile order, across the rows a tile lays side by side, then down `K`. A stored tile is read whole
/// by as many units as reads it holds down `K`, every transaction of it in flight at once; it sizes
/// no plane where no tile is stored, nor against a float rhs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneShuffleKSplit {
    /// The whole plane: one run of columns a plane, its units reading `K` together.
    WholePlane,
    /// As many units as read one run of the rhs whole.
    Run(Run),
}

impl PlaneShuffleKSplit {
    /// Units along `K` on `hw`, a plane `plane_width` units wide, for a block `depth` reads deep of
    /// `item`'s float rhs, which is stored in rows: a unit's step is one read of the run, and the
    /// plane's steps the window it runs through. A divisor of the plane, since the rest of it lies
    /// along `N`.
    pub fn units(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        depth: NonZeroUsize,
        plane_width: usize,
    ) -> usize {
        let wanted = match self {
            Self::WholePlane => plane_width,
            Self::Run(run) => Self::steps_down_k(run, hw, item, depth, plane_width),
        };
        (1..=wanted.clamp(1, plane_width))
            .rev()
            .find(|units| plane_width.is_multiple_of(*units))
            .unwrap_or(1)
    }

    /// Steps down `K` one `run` of `item`'s float rhs holds, a step `depth` reads deep, over the
    /// window a plane `plane_width` units wide reads: one where the rhs runs across `K`.
    fn steps_down_k(
        run: Run,
        hw: &Hardware,
        item: &MatmulWorkItem,
        depth: NonZeroUsize,
        plane_width: usize,
    ) -> usize {
        let element = item.elements.rhs;
        let step = depth.get() * hw.vector_ceiling(element);
        let MatmulDimensions { n, k, .. } = item.dimensions;
        let (read, plane) = match item.layouts.rhs {
            MatmulLayout::Col => (
                VectorTile::run(k, step),
                Space::new(&[(n, 1), (k, step * plane_width)]),
            ),
            MatmulLayout::Row | MatmulLayout::Strided => (
                VectorTile::run(n, step),
                Space::new(&[(k, plane_width), (n, step)]),
            ),
        };
        read.run_loads(run, &plane, element.size_bits(), hw.transaction_bytes())
            .map_or(1, |steps| {
                steps
                    .into_iter()
                    .filter(|&(axis, _)| axis == k)
                    .map(|(_, steps)| steps)
                    .product()
            })
    }
}

impl core::fmt::Display for PlaneShuffleKSplit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::WholePlane => "whole-plane",
            Self::Run(Run::Load) => "one-load",
            Self::Run(Run::Transaction) => "one-transaction",
            Self::Run(Run::StoredTile) => "one-tile",
        })
    }
}
