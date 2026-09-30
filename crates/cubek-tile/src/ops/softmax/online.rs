//! The online softmax as a tile op: a score folded into its rows' running max and sum one block
//! of its reduced axis at a time, where its holder already keeps the score.

use cubecl::prelude::*;

use crate::*;

/// The running max `m` and sum `l` of every row of a score its holder owns, in registers.
///
/// Which rows are owned is read off the score, never stated by the caller: a unit's register
/// block holds whole rows of its own, and a plane's window of shared memory is the plane's, every
/// unit of it keeping the state of every row and meeting the others on each row's partials.
///
/// A row is a score's cells along its innermost axis, the one the softmax reduces; every other
/// axis indexes the rows, so a group of query heads and their positions are rows alike.
#[derive(CubeType)]
pub struct OnlineSoftmax<E: Float> {
    m: Array<E>,
    l: Array<E>,
    #[cube(comptime)]
    rows: usize,
}

/// A score's rows: its cells along every axis but `reduced`, its innermost, which each row runs
/// along. The reduced axis may be one a walk has yet to cut, whose extent only the launch knows.
fn rows(space: &Space, reduced: Axis) -> usize {
    let rank = space.rank();
    assert!(
        space.axis_at(rank - 1) == reduced,
        "OnlineSoftmax: a row runs along the score's innermost axis, and {reduced:?} is not \
         the innermost of {space:?}"
    );
    (0..rank - 1).map(|p| space.extent_at(p)).product()
}

/// A score's rows and the extent of the axis each row runs along, its innermost.
fn rows_and_columns(space: &Space) -> (usize, usize) {
    let reduced = space.axis_at(space.rank() - 1);
    (rows(space, reduced), space.extent(reduced))
}

#[cube]
impl<E: Float> OnlineSoftmax<E> {
    /// The state of every row of `score` reduced along `reduced`, its innermost axis, before any
    /// block is folded.
    pub fn over<S: Numeric>(score: &Tile<S>, #[comptime] reduced: Axis) -> OnlineSoftmax<E> {
        OnlineSoftmax::<E>::new(comptime!(rows(&score.place.space, reduced)))
    }

    /// The state of `rows` rows before any block is folded: what a holder opens before the walk
    /// that hands it its score, as a cube's plane does before the cube's ring.
    pub fn new(#[comptime] rows: usize) -> OnlineSoftmax<E> {
        let mut m = Array::<E>::new(rows);
        let mut l = Array::<E>::new(rows);
        #[unroll]
        for r in 0..rows {
            m[r] = E::min_value();
            l[r] = E::from_int(0);
        }
        OnlineSoftmax::<E> { m, l, rows }
    }

    /// Fold one block of `score` into the rows. The block becomes `score · scale + bias`, the rows'
    /// max and sum move on, and `score` is left holding `exp(score − max)`: the block's
    /// probabilities, unnormalized. Returns each row's `exp(max before − max after)`, what anything
    /// summed against the earlier blocks is multiplied by ([`Tile::mul_rows`]).
    ///
    /// `bias` is a procedural tile read at `score`'s own coordinates, windowed to the same region.
    /// A masked cell's bias is `E::min_value()`; a row whose every cell so far is masked holds
    /// zeros and a zero sum. On a plane's window every unit of the plane calls it, and the caller
    /// meets the plane before it and before the probabilities are read.
    pub fn step(&mut self, score: &mut Tile<E>, bias: &Tile<E>, scale: E) -> Array<E> {
        let space = comptime!(score.place.space.clone());
        let holder = comptime!(score.place.holder());
        let procedural = recipe_of(bias);
        let mut max = Array::<E>::new(comptime!(self.rows));
        let mut sum = Array::<E>::new(comptime!(self.rows));
        match &mut score.kind {
            TileKind::PlanePartition(partition) => {
                let mut tile = partition.fragment();
                self.step_held(
                    &mut tile,
                    &procedural,
                    scale,
                    space.clone(),
                    &mut max,
                    &mut sum,
                );
            }
            TileKind::PlaneTile(tile) => {
                self.step_held(tile, &procedural, scale, space.clone(), &mut max, &mut sum);
            }
            TileKind::Memory(window) => {
                comptime!(assert!(
                    holder == ComputeScope::Plane,
                    "OnlineSoftmax::step: a window of shared memory is folded by the plane that \
                     holds it; this one is held by {holder:?}"
                ));
                self.bias_window(window, &procedural, scale, space.clone());
                self.fold_window(window, &mut max, &mut sum, space.clone());
            }
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => panic!(
                "OnlineSoftmax::step: a score is folded in a unit's registers or in a plane's \
                 window of shared memory"
            ),
        }
        self.advance(&max, &sum)
    }

    /// [`step`](Self::step) over a tile its units hold: a register block a unit owns whole.
    fn step_held(
        &self,
        tile: &mut PlaneTile<E>,
        bias: &Procedural<E>,
        scale: E,
        #[comptime] space: Space,
        max: &mut Array<E>,
        sum: &mut Array<E>,
    ) {
        match tile {
            PlaneTile::Registers(block) => {
                self.bias_registers(block, bias, scale, space);
                self.fold_registers(block, max, sum);
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                "OnlineSoftmax::step: a fragment's rows lie across its plane's units in the \
                 instruction's own layout; drain it into its plane's window first"
            ),
        }
    }

    /// Each row's `1 / l`, what its sum of probabilities times values is normalized by; zero for
    /// a row whose every cell was masked.
    pub fn recip_l(&self) -> Array<E> {
        let rows = comptime!(self.rows);
        let mut recip = Array::<E>::new(rows);
        #[unroll]
        for r in 0..rows {
            let live = self.l[r] > E::from_int(0);
            recip[r] = select(live, self.l[r].recip(), E::from_int(0));
        }
        recip
    }

    /// Move the rows on by one block's maxes and sums, returning each row's correction.
    fn advance(&mut self, max: &Array<E>, sum: &Array<E>) -> Array<E> {
        let rows = comptime!(self.rows);
        let mut correction = Array::<E>::new(rows);
        #[unroll]
        for r in 0..rows {
            correction[r] = (self.m[r] - max[r]).exp();
            self.l[r] = correction[r] * self.l[r] + sum[r];
            self.m[r] = max[r];
        }
        correction
    }

    /// `block = block · scale + bias`, cell by cell of a unit's register block.
    fn bias_registers(
        &self,
        block: &mut RegisterData<E>,
        bias: &Procedural<E>,
        scale: E,
        #[comptime] space: Space,
    ) {
        let bias_space = comptime!(bias.space.clone());
        comptime!(assert!(
            block.fold == 1,
            "OnlineSoftmax::step: a register block whose lines hold partials of one cell is \
             folded before its rows are read"
        ));
        let width = comptime!(block.vector_size);
        let lines = comptime!(block.nr);
        #[unroll]
        for r in 0..comptime!(block.mr) {
            #[unroll]
            for n in 0..lines {
                let at = comptime!(r * lines + n);
                let mut line = block.data[at];
                #[unroll]
                for j in 0..width {
                    let coords = score_coords(
                        (r as u32).runtime(),
                        comptime!((n * width + j) as u32).runtime(),
                        comptime!(space.clone()),
                        comptime!(bias_space.clone()),
                    );
                    let value = line.extract(j) * scale
                        + bias.evaluate(&coords, comptime!(bias_space.clone()));
                    line.insert(j, value);
                }
                block.data[at] = line;
            }
        }
    }

    /// Each row's max over this block and the earlier ones, then `exp(cell − max)` in place and
    /// each row's sum of it, over a unit's register block.
    fn fold_registers(&self, block: &mut RegisterData<E>, max: &mut Array<E>, sum: &mut Array<E>) {
        let width = comptime!(block.vector_size);
        let lines = comptime!(block.nr);
        #[unroll]
        for r in 0..comptime!(block.mr) {
            let mut row_max = self.m[r];
            #[unroll]
            for n in 0..lines {
                let line = block.data[comptime!(r * lines + n)];
                #[unroll]
                for j in 0..width {
                    row_max = cubecl::prelude::max(row_max, line.extract(j));
                }
            }
            max[r] = row_max;
            let mut row_sum = E::from_int(0);
            #[unroll]
            for n in 0..lines {
                let at = comptime!(r * lines + n);
                let mut line = block.data[at];
                #[unroll]
                for j in 0..width {
                    let p = probability::<E>(line.extract(j), row_max);
                    line.insert(j, p);
                    row_sum += p;
                }
                block.data[at] = line;
            }
            sum[r] = row_sum;
        }
    }

    /// `window = window · scale + bias`, the plane's units taking each row's cells in turns.
    fn bias_window(
        &self,
        window: &mut Memory<E>,
        bias: &Procedural<E>,
        scale: E,
        #[comptime] space: Space,
    ) {
        let (rows, columns) = comptime!(rows_and_columns(&space));
        let bias_space = comptime!(bias.space.clone());
        let size!(W) = 1usize;
        let mut view = window.flat_mut::<W>();
        let workers = PLANE_DIM as usize;
        #[unroll]
        for r in 0..rows {
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                let at = r * columns + c;
                let coords = score_coords(
                    (r as u32).runtime(),
                    c as u32,
                    comptime!(space.clone()),
                    comptime!(bias_space.clone()),
                );
                let value = view.read(at).extract(0usize) * scale
                    + bias.evaluate(&coords, comptime!(bias_space.clone()));
                view.write(at, Vector::cast_from(value));
                c += workers;
            }
        }
    }

    /// [`fold_registers`](Self::fold_registers) over a plane's window, each row's partials met
    /// over the plane.
    fn fold_window(
        &self,
        window: &mut Memory<E>,
        max: &mut Array<E>,
        sum: &mut Array<E>,
        #[comptime] space: Space,
    ) {
        let (rows, columns) = comptime!(rows_and_columns(&space));
        let size!(W) = 1usize;
        let mut view = window.flat_mut::<W>();
        let workers = PLANE_DIM as usize;
        #[unroll]
        for r in 0..rows {
            let mut partial = self.m[r];
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                partial = cubecl::prelude::max(partial, view.read(r * columns + c).extract(0usize));
                c += workers;
            }
            let row_max = plane_max(partial);
            max[r] = row_max;
            let mut partial = E::from_int(0);
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                let at = r * columns + c;
                let p = probability::<E>(view.read(at).extract(0usize), row_max);
                view.write(at, Vector::cast_from(p));
                partial += p;
                c += workers;
            }
            sum[r] = plane_sum(partial);
        }
    }
}

/// `exp(value − max)`, zero where the row's max is itself masked: a row with nothing live yet.
#[cube]
fn probability<E: Float>(value: E, row_max: E) -> E {
    let live = row_max > E::min_value();
    select(live, (value - row_max).exp(), E::from_int(0))
}

/// The coordinates in `bias_space` of cell `column` of row `row` of a score over `space`: the row
/// unravelled over every axis but the innermost, the column along the innermost, and an axis the
/// score does not span at zero.
#[cube]
fn score_coords(
    row: u32,
    column: u32,
    #[comptime] space: Space,
    #[comptime] bias_space: Space,
) -> Coords<u32> {
    let rank = comptime!(space.rank());
    let row_extents = Coords::<u32>::constant(comptime!(
        (0..rank - 1)
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    ));
    let row_digits = row_extents.unravel(row);
    let mut coords = Coords::<u32>::new();
    #[unroll]
    for p in 0..comptime!(bias_space.rank()) {
        let axis = comptime!(bias_space.axis_at(p));
        if comptime!(space.axis_at(rank - 1) == axis) {
            coords.push(column);
        } else if comptime!(space.contains(axis)) {
            coords.push(row_digits.at(comptime!(space.position(axis))));
        } else {
            coords.push(0u32.runtime());
        }
    }
    coords
}

/// The recipe a bias tile evaluates: a bias is procedural, read at the score's coordinates.
#[cube]
fn recipe_of<E: Float>(bias: &Tile<E>) -> Procedural<E> {
    match &bias.kind {
        TileKind::Procedural(recipe) => recipe.clone(),
        TileKind::Memory(_)
        | TileKind::PlaneTile(_)
        | TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Lines(_) => {
            panic!("OnlineSoftmax::step: the bias is procedural, read at the score's coordinates")
        }
    }
}
