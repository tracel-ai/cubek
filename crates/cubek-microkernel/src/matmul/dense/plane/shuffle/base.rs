use crate::matmul::dense::plane::{PlaneShuffleKSplit, PlaneShuffleMicrokernelSettings};
use crate::matmul::stored::{Spread, Walked};
use crate::matmul::{
    MatmulAxes, MatmulBlocks, MatmulDim, MatmulLayout, MatmulMicrokernel, MatmulOperand,
    MatmulOperands, MatmulOrder, MatmulRows, MatmulRowsKind, MatmulStep, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Instruction, Levels, RegisterBlock};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N, blocked};
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// Bytes one stored word of a packed rhs holds: it binds one `u32` per line whatever it packs.
const WORD_BYTES: usize = core::mem::size_of::<u32>();

/// Every row of a run of the output's columns, owned by one plane: its units split `K` between
/// them in turns, each summing its share in registers, and their partials merge inside the plane
/// when the walk ends.
///
/// Where a unit holding every row would hold more sums than its registers keep, units side by
/// side along `M` split the rows: they read the same columns and the same share of `K`, so the
/// plane's loads are the same addresses, and only their sums are their own.
///
/// A unit's step is a block of the output's columns and of `K`, counted in the reads the rhs
/// is loaded in: along the axis its buffer runs a read is one vector, across it one line. So a
/// col rhs's block is one column, a run of vectors down `K` merged into its cell, and a row
/// rhs's is a few vectors of columns, each read at a few rows of `K`.
///
/// Against a packed rhs ([`PlaneShuffleBlocks`]) the units multiply each read by its block's
/// scale, and a step is the rhs's ([`MatmulStep`]). Where the scales group `K`, a unit takes
/// a word of scales a turn and the units take turns, so a unit holds whole blocks and the walk
/// over them sits above the units; where they do not, each unit takes a contiguous run of `K`. A
/// unit's columns are whole reads of the rhs, or whole words of scales where the scales group
/// `N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleMicrokernel {
    pub settings: PlaneShuffleMicrokernelSettings,
    /// Rows of the lhs one unit holds, each against every column it reads: every row of the
    /// product, or the settings' share of them.
    ///
    /// An extent a blueprint holds, an exception to the crate's rule: every row sits in registers
    /// beside each step of the rhs, so each row count is a kernel of its own whatever the
    /// binding says.
    pub rows: usize,
    /// Units side by side along `M`, each holding its own `rows`: one where a unit holds every
    /// row.
    pub units_along_m: usize,
    /// Units sharing one run of columns and rows, each taking every `units_along_k`-th step of
    /// `K`: its settings' `k_split`, resolved against the plane's share left by the units along
    /// `M`, the device's transaction size and the element.
    pub units_along_k: usize,
    /// Groups of `units_along_m × units_along_k` units the plane holds beside each other along
    /// `N`: the rest of the width the device commits to, so the groups and their units fill the
    /// plane exactly.
    pub groups_along_n: usize,
    /// The widest vector the device reads the rhs's element in, which sizes a read along the
    /// rhs's axis and the accumulator's register budget: a col rhs's cell holds a vector of
    /// partials, and the widths are settled after the microkernel, so the budget covers the widest
    /// vector it could bind. A narrower bound vector inlines inside it.
    pub rhs_vector_ceiling: usize,
    /// The order each operand is read in. The rhs's is the axis each unit's step runs along:
    /// [`InBufferOrder`](MatmulOrder::InBufferOrder) for a col rhs, contiguous along `K`,
    /// and [`Stated`](MatmulOrder::Stated) for a row rhs, contiguous along `N`; the others are
    /// read as the product names them.
    pub orders: MatmulOperands<MatmulOrder>,
    /// The caller's axes its levels are stated along.
    pub axes: MatmulAxes,
    /// The rhs's scale blocks and the step a unit moves in, where it is packed; none for a
    /// float rhs.
    pub blocks: Option<PlaneShuffleBlocks>,
}

/// A packed rhs as the plane shuffle reads it: its blocks, and the step a unit's columns and its
/// share of `K` move in.
///
/// Its levels are stated over the problem's block-split space: the digits of `N` and `K` the
/// blocks make, and the position inside a block of `M`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleBlocks {
    /// The space the rhs's blocks cut.
    pub space: MatmulBlocks,
    /// Elements of `N` one unit reads: the settings' steps of `N`, narrowed until `n` closes in
    /// them.
    ///
    /// Read off an extent, an exception to the crate's rule: the space is cut into scale blocks
    /// and compiled in whole, so a unit's columns are a cut of that space.
    pub columns: usize,
    /// Elements of `K` one unit reads a turn: one step, a word of scales where they group `K`.
    pub depth: usize,
}

impl PlaneShuffleMicrokernel {
    /// Steps of `K` a unit holds at once, which a routine reports as `CubeStaging`'s stages: the
    /// microkernel stages nothing in shared memory, and loads a step once the last is summed.
    pub const STAGES_IN_FLIGHT: usize = 1;

    /// The most rows a plane holds: every row is summed against each step of the rhs in
    /// registers, so past this many the registers, not the rhs, bound the kernel.
    pub const MAX_ROWS: usize = 8;

    /// What a product of no rows, or of more than [`MAX_ROWS`](Self::MAX_ROWS), is declined with.
    pub const ROWS_OUT_OF_RANGE: Refusal = Refusal::Declined {
        why: "the plane shuffle holds from one row to `PlaneShuffleMicrokernel::MAX_ROWS`, every one against one read of the rhs",
    };

    /// What an rhs lined along neither of its axes is declined with.
    pub const STRIDED_RHS: Refusal = Refusal::Declined {
        why: "the plane shuffle reads an rhs lined along `K` or `N`",
    };

    /// The order this microkernel reads an rhs `layout` lies as in, where it reads it: a col
    /// rhs in its buffer's order, a row rhs as the product names it, and none for an rhs
    /// lined along neither axis.
    fn order_of(layout: MatmulLayout) -> Option<MatmulOrder> {
        match layout {
            MatmulLayout::Col => Some(MatmulOrder::InBufferOrder),
            MatmulLayout::Row => Some(MatmulOrder::Stated),
            MatmulLayout::Strided => None,
        }
    }

    /// What a col rhs is declined with where it binds one element at a time: the tile DSL merges
    /// a vector of it with a vector of the lhs into one cell, so it reads both in vectors of at
    /// least two, and a `K` the steps overhang, or either buffer aligned to one element, gives
    /// none.
    pub const COL_RHS_NEEDS_VECTORS_ALONG_K: Refusal = Refusal::Declined {
        why: "the plane shuffle reads a col rhs and the lhs beside it in vectors of at least two elements along `K`",
    };

    /// What a col rhs is declined with where a unit's block holds more than one column: the tile
    /// DSL sums a col rhs's line of partials into its cell correctly only for a block of one
    /// column.
    pub const COL_RHS_ONE_COLUMN_A_UNIT: Refusal = Refusal::Declined {
        why: "the plane shuffle reads a col rhs one column a unit, each summing its line of partials",
    };

    /// Whether the operands bound at `widths` are read in vectors this microkernel merges.
    ///
    /// # Errors
    ///
    /// [`COL_RHS_NEEDS_VECTORS_ALONG_K`](Self::COL_RHS_NEEDS_VECTORS_ALONG_K), for a col rhs
    /// bound one element at a time, and
    /// [`COL_RHS_ONE_COLUMN_A_UNIT`](Self::COL_RHS_ONE_COLUMN_A_UNIT), for a col rhs under a
    /// block of more than one column.
    pub fn check_vectors(&self, widths: &MatmulOperands<usize>) -> Result<(), Refusal> {
        match (self.orders.rhs, widths.rhs, self.settings.columns.get()) {
            (MatmulOrder::InBufferOrder, 0 | 1, _) => Err(Self::COL_RHS_NEEDS_VECTORS_ALONG_K),
            (MatmulOrder::InBufferOrder, _, 2..) => Err(Self::COL_RHS_ONE_COLUMN_A_UNIT),
            (MatmulOrder::InBufferOrder | MatmulOrder::Stated, _, _) => Ok(()),
        }
    }

    /// Elements of the output's columns one unit reads a step: its block's columns, each a vector
    /// where the rhs runs along `N`, or a packed rhs's steps of them.
    pub fn columns(&self) -> usize {
        if let Some(blocks) = &self.blocks {
            return blocks.columns;
        }
        self.settings.columns.get()
            * match self.orders.rhs {
                MatmulOrder::InBufferOrder => 1,
                MatmulOrder::Stated => self.rhs_vector_ceiling,
            }
    }

    /// Elements of `K` one unit reads a step: its block's depth, each a vector where the rhs
    /// runs along `K`, or a packed rhs's step of it.
    pub fn depth(&self) -> usize {
        if let Some(blocks) = &self.blocks {
            return blocks.depth;
        }
        self.settings.depth.get()
            * match self.orders.rhs {
                MatmulOrder::InBufferOrder => self.rhs_vector_ceiling,
                MatmulOrder::Stated => 1,
            }
    }

    /// The turns the plane's units take through a packed rhs's `K`, a word of scale blocks a
    /// unit a turn, which a cube's planes divide between them. `None` for a float rhs, and
    /// where the scales group no `K`.
    pub fn turns(&self) -> Option<usize> {
        let space = &self.blocks?.space;
        space.operand(MatmulOperand::Rhs).grouping(K, space.k)?;
        let [(_, blocks), _] = space.operand(MatmulOperand::Rhs).axes(K, space.k);
        Some(blocks / (space.operand(MatmulOperand::Rhs).scales_per_word() * self.units_along_k))
    }

    /// The plane width this kernel is compiled for, which its launch must match: the units add
    /// their partials across exactly this many lanes.
    pub fn plane_width(&self) -> usize {
        self.groups_along_n * self.units_along_m * self.units_along_k
    }

    /// The levels inside one plane, which a routine states the rest of the partitioning around:
    /// [`step_levels`](Self::step_levels), then the walk through all of `K`, above the units where
    /// they take `K` in turns, below them where each takes a run of a packed rhs's `K`.
    pub fn tiling(&self) -> Levels {
        let Some(blocks) = &self.blocks else {
            return self.float_step_levels().walk_every(&[self.axes.k]);
        };
        let spread = self.spread(blocks);
        match spread.walk {
            Walked::Above(_) => self.in_turns(blocks, &spread).walk_every(&[blocked(K).0]),
            Walked::Below(axis, steps) => self.in_runs(&spread, (axis, steps)),
        }
    }

    /// What one step of the plane covers where its units take `K` in turns: one unit's step, and
    /// the plane's units over `K` in turns, over `M` and over `N` beside each other. A cube whose
    /// planes share a step states them around this. `None` where each unit walks a run of a packed
    /// rhs's `K` of its own, which planes cannot split without splitting the run.
    pub fn step_levels(&self) -> Option<Levels> {
        let Some(blocks) = &self.blocks else {
            return Some(self.float_step_levels());
        };
        let spread = self.spread(blocks);
        matches!(spread.walk, Walked::Above(_)).then(|| self.in_turns(blocks, &spread))
    }

    /// [`step_levels`](Self::step_levels) for a float rhs.
    ///
    /// An axis the units do not spread over is not stated: the tile DSL reads a stated level of one
    /// as a level the accumulator has to shape itself to.
    fn float_step_levels(&self) -> Levels {
        let units_along_k = self.units_along_k;
        let MatmulAxes { m, n, k } = self.axes;
        let leaf = Levels::leaf(&[(m, self.rows), (n, self.columns()), (k, self.depth())]);
        let units: Vec<_> = [
            (n, self.groups_along_n),
            (m, self.units_along_m),
            (k, units_along_k),
        ]
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .collect();
        // Stated even when it cuts nothing, on a plane of one unit: the walk steps into it.
        match units_along_k {
            1 => leaf.units(&units),
            _ => leaf.units(&units).interleaved(k),
        }
    }

    /// The columns a unit reads of a packed rhs, whose blocks cut `space`, and the units along
    /// `K` that spread it, for each count of steps of `N` ([`MatmulStep`]) from the settings' down
    /// to one whose columns close `n` in one digit and leave `shared` units, the plane's share
    /// left by the units along `M`, whole groups along `N`.
    ///
    /// A unit reads one step of `K` a turn by the whole share, the units as many as the scales'
    /// words divide: where the units take a word of scales a turn, a turn is a step along `K`, and
    /// a step a load makes deeper than a word of scales is declined. Settings stating another
    /// depth or split of `K` size none.
    fn packed_steps(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneShuffleMicrokernelSettings,
        space: MatmulBlocks,
        shared: usize,
    ) -> impl Iterator<Item = (usize, PlaneShuffleBlocks)> {
        let operand = space.operand(MatmulOperand::Rhs);
        let step = MatmulStep::new(hw, space.operands(), item.elements);
        let (depth, columns_a_step) = (step.along(K), step.along(N));
        let units_along_k = operand.units_distributing(K, space.k, shared);
        let groups_along_n = shared / units_along_k;
        let one_step_a_turn = settings.depth == NonZeroUsize::MIN
            && settings.k_split == PlaneShuffleKSplit::WholePlane
            && operand
                .grouping(K, space.k)
                .is_none_or(|block| depth == block * operand.scales_per_word())
            && operand.spread_across(K, space.k, units_along_k).is_ok();
        let steps = match one_step_a_turn {
            true => settings.columns.get(),
            false => 0,
        };
        (1..=steps)
            .rev()
            .map(move |steps| steps * columns_a_step)
            .filter(move |&columns| {
                space.n.is_multiple_of(columns)
                    && (space.n / columns).is_multiple_of(groups_along_n)
                    && operand.cut(N, space.n, columns).len() == 1
            })
            .map(move |columns| {
                (
                    units_along_k,
                    PlaneShuffleBlocks {
                        space,
                        columns,
                        depth,
                    },
                )
            })
    }

    /// The axis a unit's columns land on, and how many of it a unit holds: the digit of `N` a
    /// packed rhs's blocks make it.
    pub fn column_cut(&self) -> (Axis, usize) {
        let Some(blocks) = &self.blocks else {
            return (self.axes.n, self.columns());
        };
        let [cut] = blocks
            .space
            .operand(MatmulOperand::Rhs)
            .cut(N, blocks.space.n, blocks.columns)[..]
        else {
            unreachable!("candidates hold the columns to one digit");
        };
        cut
    }

    /// The cut of `M` a unit's rows land on in a packed rhs's space, where units along `M`
    /// split them.
    fn row_cut(&self) -> Option<(Axis, usize)> {
        (self.units_along_m > 1).then(|| (blocked(M).1, self.rows))
    }

    /// The unit's step and the plane's units, the units taking `K`'s scale blocks in turns.
    fn in_turns(&self, blocks: &PlaneShuffleBlocks, spread: &Spread) -> Levels {
        let space = &blocks.space;
        let along_k = space
            .operand(MatmulOperand::Rhs)
            .span(K, space.k, blocks.depth);
        let leaf = Levels::leaf(
            &[self.column_cut()]
                .into_iter()
                .chain(self.row_cut())
                .chain(along_k)
                .collect::<Vec<_>>(),
        );
        spread
            .interleaved
            .iter()
            .fold(leaf.units(&self.block_units(spread)), |tiling, &axis| {
                tiling.interleaved(axis)
            })
    }

    /// The unit's block walked `walk` below the plane's units, each unit taking a run of `K`.
    fn in_runs(&self, spread: &Spread, walk: (Axis, usize)) -> Levels {
        let leaf = Levels::leaf(
            &[self.column_cut()]
                .into_iter()
                .chain(self.row_cut())
                .chain(spread.leaf.iter().copied())
                .collect::<Vec<_>>(),
        );
        leaf.walk(&[walk]).units(&self.block_units(spread))
    }

    /// How the plane's units spread a packed rhs's `K`.
    fn spread(&self, blocks: &PlaneShuffleBlocks) -> Spread {
        let space = &blocks.space;
        space
            .operand(MatmulOperand::Rhs)
            .spread_across(K, space.k, self.units_along_k)
            .expect("candidates hold the units to a spread of K")
    }

    /// The plane's units over a packed rhs's space: its groups side by side along `N`, the
    /// units of each along `M`, and along `K`.
    fn block_units(&self, spread: &Spread) -> Vec<(Axis, usize)> {
        [(self.column_cut().0, self.groups_along_n)]
            .into_iter()
            .chain(self.row_cut().map(|(axis, _)| (axis, self.units_along_m)))
            .chain(spread.units.iter().copied())
            .collect()
    }

    /// The extent along `dim` of one unit's step: the leaf of [`tiling`](Self::tiling).
    pub fn leaf_extent(&self, dim: MatmulDim) -> usize {
        match dim {
            MatmulDim::M => self.rows,
            MatmulDim::N => self.columns(),
            MatmulDim::K => self.depth(),
        }
    }

    /// 32-bit registers one unit keeps: its accumulator, summed in `item.sum`, and one step's
    /// block of the lhs and the rhs, a packed rhs's a word a column. The compiler adds
    /// addressing and loop state on top.
    pub fn registers_per_unit(&self, item: &MatmulWorkItem) -> u32 {
        let [rows, columns] = [self.rows, self.columns()];
        let (lhs, rhs) = match &self.blocks {
            None => (
                rows * self.depth(),
                columns * self.depth() * item.elements.rhs.size(),
            ),
            Some(blocks) => (
                rows * blocks.space.operand(MatmulOperand::Rhs).read(K),
                columns * WORD_BYTES,
            ),
        };
        let bytes =
            self.accumulator_scalars() * item.sum.size() + lhs * item.elements.lhs.size() + rhs;
        bytes.div_ceil(4) as u32
    }

    /// Scalars one unit's accumulator holds: a cell per row and column, which for a col rhs is
    /// a line of partials summed on drain, a packed rhs's a read of its values along `K`.
    fn accumulator_scalars(&self) -> usize {
        let cell = match (&self.blocks, self.orders.rhs) {
            (Some(blocks), _) => blocks.space.operand(MatmulOperand::Rhs).read(K),
            (None, MatmulOrder::InBufferOrder) => self.rhs_vector_ceiling,
            (None, MatmulOrder::Stated) => 1,
        };
        self.rows * self.columns() * cell
    }

    /// The instruction a plane's sum opens on: a register block holding all of one unit's
    /// accumulator.
    pub fn tile_instruction(&self) -> Instruction {
        Instruction::Registers {
            config: RegisterBlock::new(self.accumulator_scalars()),
        }
    }
}

/// Its rows, its settings, and how the rhs lies.
impl core::fmt::Display for PlaneShuffleMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let weight = match self.orders.rhs {
            MatmulOrder::InBufferOrder => "col",
            MatmulOrder::Stated => "row",
        };
        write!(f, "rows={} {} rhs={weight}", self.rows, self.settings)
    }
}

impl Microkernel for PlaneShuffleMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = PlaneShuffleMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = ();

    const DECLINE: &'static str = "the plane shuffle needs a plane width the device commits to, and against a packed rhs unit columns that close n";

    /// The default, then one axis off it at a time, then the blocks a star never reaches: deep
    /// reads by units that own their columns, the row rhs's wide blocks, and the default at two
    /// rows a unit. Sixteen columns is the accumulator's bound, not the device's: at eight rows a
    /// wider block outgrows a unit's registers and spills, unless units split the rows.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::ONE_READ,
            Self::Settings::TWO_COLUMNS,
            Self::Settings::ONE_TRANSACTION,
            Self::Settings::EIGHT_READS_ONE_TRANSACTION,
            Self::Settings::ONE_READ_ONE_TRANSACTION,
            Self::Settings::TWO_ROW_VECTORS,
            Self::Settings::FOUR_ROW_VECTORS,
            Self::Settings::TWO_ROWS_A_UNIT,
        ]
    }

    /// From one row to [`MAX_ROWS`](Self::MAX_ROWS), every one held whole, and an rhs lined
    /// along one of its axes.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.rows {
            MatmulRows::Whole(rows) if (1..=Self::MAX_ROWS).contains(&rows) => {}
            MatmulRows::Whole(_) | MatmulRows::Tiled => return Err(Self::ROWS_OUT_OF_RANGE),
        }
        match item.layouts.rhs {
            MatmulLayout::Col | MatmulLayout::Row => Ok(()),
            MatmulLayout::Strided => Err(Self::STRIDED_RHS),
        }
    }

    /// The one the settings size, where the device commits to a plane width: its units merge their
    /// partials across the plane, so a guessed width sums the wrong units. It holds every row of
    /// the product, one to [`MAX_ROWS`](Self::MAX_ROWS), and reads an rhs lined along one of its
    /// axes. Against a packed rhs, one for each count of steps of `N` from the settings' down
    /// to one (its packed steps).
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let rows = match item.rows {
            MatmulRows::Whole(rows) if (1..=Self::MAX_ROWS).contains(&rows) => Some(rows),
            MatmulRows::Whole(_) | MatmulRows::Tiled => None,
        };
        let rhs = Self::order_of(item.layouts.rhs);
        let plane_width = hw.plane_size().ok().map(|width| width as usize);
        let Some(((rows, rhs), plane_width)) = rows.zip(rhs).zip(plane_width) else {
            return Vec::new().into_iter();
        };
        let units_along_m = settings.units_along_m(rows, plane_width);
        // The plane's share left by the units along `M`, for the units along `K` and `N`.
        let shared = plane_width / units_along_m;
        let sized = |units_along_k: usize, blocks| Self {
            settings: *settings,
            rows: rows / units_along_m,
            units_along_m,
            units_along_k,
            groups_along_n: shared / units_along_k,
            rhs_vector_ceiling: hw.vector_ceiling(item.elements.rhs),
            orders: MatmulOperands {
                rhs,
                ..MatmulOperands::all(MatmulOrder::Stated)
            },
            axes: item.axes,
            blocks,
        };
        match item.blocks {
            None => {
                let units_along_k = settings.k_split.units(hw, item, settings.depth, shared);
                vec![sized(units_along_k, None)]
            }
            Some(space) => Self::packed_steps(hw, item, settings, space, shared)
                .map(|(units_along_k, blocks)| sized(units_along_k, Some(blocks)))
                .collect(),
        }
        .into_iter()
    }

    /// Each operand's step is the leaf's extent along the axis it is bound innermost along: the lhs
    /// down `K`, the rhs along its own contiguous axis, the output along `N`. A col rhs's
    /// step is contracted with the lhs's, a line of each at a time; a row rhs's lands in the
    /// output's cells, lined with it.
    ///
    /// Nothing against a packed rhs: every width follows from its packing, and a count of a
    /// float element's vectors means nothing for words.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        if self.blocks.is_some() {
            return;
        }
        for operand in MatmulOperand::ALL {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: self.leaf_extent(self.orders.innermost(operand)),
            });
        }
        let rhs_partner = match self.orders.rhs {
            MatmulOrder::InBufferOrder => MatmulOperand::Lhs,
            MatmulOrder::Stated => MatmulOperand::Out,
        };
        vectorization.require(VectorRequirement::Together {
            operand: MatmulOperand::Rhs,
            with: rhs_partner,
        });
    }

    /// It holds its sums in registers and contracts them with the device's vector arithmetic: the
    /// work. The reads are its routine's to count, which places its units.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: self.registers_per_unit(item),
        });
    }
}

impl MatmulMicrokernel for PlaneShuffleMicrokernel {
    /// It holds every row, compiled in, against each step of the rhs.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.orders
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::stored::Operand;
    use crate::matmul::stored::Packing;
    use crate::matmul::tile_order;
    use crate::matmul::{MatmulAxes, MatmulDestination};
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `rows` rows of `f16` against a col rhs, every row held whole.
    fn item(rows: usize) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: MatmulOperands::all(F16),
            sum: F32,
            layouts: MatmulOperands {
                rhs: MatmulLayout::Col,
                ..MatmulOperands::all(MatmulLayout::Row)
            },
            blocks: None,
            rows: MatmulRows::Whole(rows),
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::MNK,
        }
    }

    /// The plane at `rows` rows and `settings`, on a 32-wide plane.
    fn plane(rows: usize, settings: PlaneShuffleMicrokernelSettings) -> PlaneShuffleMicrokernel {
        PlaneShuffleMicrokernel::candidates(&devices::m2_pro(), &item(rows), &settings)
            .next()
            .expect("the plane is sized")
    }

    /// Eight rows at two a unit: four units along `M`, which leave eight of the plane's units to
    /// share `K`, and the plane still filled.
    #[test]
    fn units_split_eight_rows_two_a_unit() {
        let plane = plane(8, PlaneShuffleMicrokernelSettings::TWO_ROWS_A_UNIT);
        assert_eq!(
            (plane.rows, plane.units_along_m, plane.units_along_k),
            (2, 4, 8)
        );
        assert_eq!(plane.plane_width(), 32);
    }

    /// Rows a unit's share cannot split stay whole: one row, and three rows the plane's width does
    /// not divide.
    #[test]
    fn rows_no_unit_split_divides_stay_whole() {
        let one_a_unit = PlaneShuffleMicrokernelSettings {
            rows_per_unit: NonZeroUsize::new(1),
            ..PlaneShuffleMicrokernelSettings::DEFAULT
        };
        assert_eq!(plane(1, one_a_unit).units_along_m, 1);
        assert_eq!(plane(3, one_a_unit).units_along_m, 1);
    }

    /// Split rows cut a unit's registers: a quarter of the accumulator at eight rows, two a unit.
    #[test]
    fn split_rows_hold_fewer_sums_a_unit() {
        let whole = plane(8, PlaneShuffleMicrokernelSettings::DEFAULT);
        let split = plane(8, PlaneShuffleMicrokernelSettings::TWO_ROWS_A_UNIT);
        assert_eq!(whole.accumulator_scalars(), 4 * split.accumulator_scalars());
    }

    /// `rows` rows against a `4096 × 4096` NVFP4 rhs in tile order, read at four words a load:
    /// values a block down `k` a read, scales a word of four blocks.
    fn tile_order(rows: usize) -> MatmulWorkItem {
        let rhs = Operand::packed(K, 8, K, tile_order::BLOCK, 4)
            .stored(
                N,
                4,
                Some((
                    &tile_order::values_axes(N),
                    tile_order::values_storage(N, (128, 8), 4).tiles(),
                )),
                None,
            )
            .expect("tile order is read");
        MatmulWorkItem {
            elements: MatmulOperands {
                lhs: F16,
                rhs: F32,
                out: F16,
            },
            sum: F32,
            layouts: MatmulOperands {
                rhs: MatmulLayout::Col,
                ..MatmulOperands::all(MatmulLayout::Row)
            },
            blocks: Some(MatmulBlocks {
                packings: MatmulOperands {
                    rhs: Some(Packing {
                        operand: rhs,
                        scheme: tile_order::scheme(),
                    }),
                    ..MatmulOperands::all(None)
                },
                m: rows,
                n: 4096,
                k: 4096,
            }),
            rows: MatmulRows::Whole(rows),
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::INSIDE_BLOCKS,
        }
    }

    /// The lead plane against the packed rhs at `rows` rows and `rows_per_unit` rows a unit,
    /// on a 32-wide plane.
    fn packed(rows: usize, rows_per_unit: usize) -> PlaneShuffleMicrokernel {
        let settings = PlaneShuffleMicrokernelSettings::stepping(1, Some(rows_per_unit));
        PlaneShuffleMicrokernel::candidates(&devices::m2_pro(), &tile_order(rows), &settings)
            .next()
            .expect("the lead plane is sized")
    }

    /// Eight rows at two a unit against a packed rhs: four units along `M`, which leave eight
    /// of the plane's units to share `K`; the step is one column by a word of scales, sixty-four
    /// of `k`.
    #[test]
    fn units_split_eight_rows_of_a_packed_product_two_a_unit() {
        let plane = packed(8, 2);
        assert_eq!(
            (plane.rows, plane.units_along_m, plane.units_along_k),
            (2, 4, 8)
        );
        assert_eq!(plane.plane_width(), 32);
        assert_eq!((plane.columns(), plane.depth()), (1, 64));
    }

    /// Rows a unit's share cannot split stay whole against a packed rhs too: one row, and three
    /// rows the plane's width does not divide.
    #[test]
    fn packed_rows_no_unit_split_divides_stay_whole() {
        assert_eq!(packed(1, 2).units_along_m, 1);
        assert_eq!(packed(3, 1).units_along_m, 1);
    }

    /// A packed rhs is read a step of `K` a turn by the whole plane: a setting stating a deeper
    /// block, or another split of `K`, sizes no plane against it.
    #[test]
    fn a_packed_rhs_is_read_one_step_a_turn() {
        for settings in [
            PlaneShuffleMicrokernelSettings::DEFAULT,
            PlaneShuffleMicrokernelSettings::ONE_READ_ONE_TRANSACTION,
        ] {
            assert_eq!(
                PlaneShuffleMicrokernel::candidates(&devices::m2_pro(), &tile_order(1), &settings)
                    .count(),
                0,
                "{settings}"
            );
        }
    }
}
