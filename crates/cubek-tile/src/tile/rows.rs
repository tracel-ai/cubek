//! A tile's rows where its holder keeps them ([`Rows`]).

use cubecl::prelude::*;

use crate::*;

/// Who keeps a tile's rows, and so how a row is read whole.
#[allow(clippy::large_enum_variant)]
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum RowsKind<E: Float> {
    /// A register block a unit holds whole: a row is in its registers.
    Registers(RegisterData<E>),
    /// A window of shared memory a plane holds: its units take a row's cells in turns and meet on
    /// the row's partials.
    Window(Memory<E>),
    /// A plane's grid of cmma fragments, whose rows lie across its units in the instruction's own
    /// layout: reached only to be scaled, through the plane's scratch.
    Fragments(PlanePartition<E>),
    /// One cmma fragment of a plane's, reached the same way.
    Fragment(PlaneTile<E>),
}

/// A tile's rows, read and written where its holder keeps them, whoever that is: the verbs a
/// row-wise op runs, each over every row, and each row whole.
///
/// A row is the tile's cells along its innermost axis; every other axis indexes the rows, so a
/// group of query heads and their positions are rows alike. The rows write through to the tile.
#[derive(CubeType)]
pub struct Rows<E: Float> {
    kind: RowsKind<E>,
    #[cube(comptime)]
    space: Space,
}

#[cube]
impl<E: Float> Tile<E> {
    /// This tile's rows, where its holder keeps them: a unit's register block, a plane's window of
    /// shared memory, or a plane's fragments. A window of memory any wider holder shares has no
    /// one to read its rows whole, and is refused.
    pub fn rows(&self) -> Rows<E> {
        let holder = comptime!(self.place.holder());
        let kind = match &self.kind {
            TileKind::PlanePartition(partition) => partition.rows(),
            TileKind::PlaneTile(tile) => tile.rows(),
            TileKind::Memory(window) => {
                comptime!(assert!(
                    holder == ComputeScope::Plane,
                    "Tile::rows: a window of shared memory is read row by row by the plane that \
                     holds it; this one is held by {holder:?}"
                ));
                RowsKind::new_Window(window.clone())
            }
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => panic!(
                "Tile::rows: a register block, a plane's fragments, or a plane's window holds rows"
            ),
        };
        Rows::<E> {
            kind,
            space: comptime!(self.place.space.clone()),
        }
    }
}

#[cube]
impl<E: Float> PlanePartition<E> {
    /// This grid's rows: a unit's register block where the grid is one, the fragments otherwise.
    pub(crate) fn rows(&self) -> RowsKind<E> {
        match self.at(0usize, 0usize) {
            PlaneTile::Registers(block) => {
                comptime!(assert!(
                    self.m_tiles * self.n_tiles == 1,
                    "Tile::rows: a unit's rows are one register block"
                ));
                RowsKind::new_Registers(block)
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => RowsKind::new_Fragments(self.clone()),
        }
    }
}

#[cube]
impl<E: Float> PlaneTile<E> {
    /// This tile's rows: a register block's, or one fragment's.
    pub(crate) fn rows(&self) -> RowsKind<E> {
        match self {
            PlaneTile::Registers(block) => RowsKind::new_Registers(block.clone()),
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => RowsKind::new_Fragment(self.clone()),
        }
    }
}

#[cube]
impl<E: Float> Rows<E> {
    /// `self[r, c] = self[r, c] · scale + bias[r, c]`, the procedural `bias` read at each cell's
    /// coordinates, windowed to the same region as the tile.
    pub(crate) fn scale_add(&mut self, scale: E, bias: &Tile<E>) {
        let recipe = bias.recipe();
        match &mut self.kind {
            RowsKind::Registers(block) => block.scale_add_rows(scale, recipe, self.space.clone()),
            RowsKind::Window(window) => window.scale_add_rows(scale, recipe, self.space.clone()),
            RowsKind::Fragments(_) | RowsKind::Fragment(_) => Rows::<E>::refuse_reading(),
        }
    }

    /// Each row's max, starting from `seed`'s.
    pub(crate) fn maxima(&self, seed: &Array<E>) -> Array<E> {
        match &self.kind {
            RowsKind::Registers(block) => block.row_maxima(seed),
            RowsKind::Window(window) => window.row_maxima(seed, self.space.clone()),
            RowsKind::Fragments(_) | RowsKind::Fragment(_) => Rows::<E>::refuse_reading(),
        }
    }

    /// `self[r, c] = exp(self[r, c] − rows[r])` ([`exp_minus_cell`](Self::exp_minus_cell)).
    pub(crate) fn exp_minus(&mut self, rows: &Array<E>) {
        match &mut self.kind {
            RowsKind::Registers(block) => block.exp_minus_rows(rows),
            RowsKind::Window(window) => window.exp_minus_rows(rows, self.space.clone()),
            RowsKind::Fragments(_) | RowsKind::Fragment(_) => Rows::<E>::refuse_reading(),
        }
    }

    /// Each row's sum.
    pub(crate) fn sums(&self) -> Array<E> {
        match &self.kind {
            RowsKind::Registers(block) => block.row_sums(),
            RowsKind::Window(window) => window.row_sums(self.space.clone()),
            RowsKind::Fragments(_) | RowsKind::Fragment(_) => Rows::<E>::refuse_reading(),
        }
    }

    /// `self[r, c] *= factors[r]`. A plane's fragments bounce through its scratch, met on
    /// `sync_plane`: every unit of the plane calls it.
    pub fn mul(&mut self, factors: &Array<E>) {
        match &mut self.kind {
            RowsKind::Registers(block) => block.mul_rows(factors, 0usize),
            RowsKind::Window(window) => window.mul_rows(factors, self.space.clone()),
            RowsKind::Fragments(partition) => partition.mul_rows(factors),
            RowsKind::Fragment(tile) => tile.mul_rows(factors, 0usize),
        }
    }

    /// `exp(value − row)`, zero where `row` is itself masked: a row with nothing live yet, whose
    /// max is still the minimum value.
    pub(crate) fn exp_minus_cell(value: E, row: E) -> E {
        let live = row > E::min_value();
        select(live, (value - row).exp(), E::from_int(0))
    }

    /// A fragment's rows lie across its plane's units in the instruction's own layout, so only a
    /// scale reaches them: to read a row, drain the fragments into the plane's window first.
    fn refuse_reading() -> ! {
        panic!("Rows: a fragment's rows are read after draining it into its plane's window")
    }
}
