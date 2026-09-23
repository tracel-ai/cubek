//! What a contraction reads as one factor of its terms: an operand's values, and the scales that
//! multiply them.
//!
//! A quantized tensor is values *and* scales, and an operand's binding names one thing, so the
//! scales are an operand of their own and folding them in is the contraction's own arithmetic.
//! Which factor they multiply, and how deep, is where and how often the kernel wrote them.
//!
//! **Nothing here multiplies anything.** `scaled` says which level a factor carries; the multiply
//! happens once per line, at the read, inside the leaf ([`ScaleLookup`]), the only place it can
//! without a dequantized tile. The verb is the contraction's ([`mm_scaled`](Tile::mm_scaled)).
//!
//! A factor that carries none is [`Tile::plain`], read with no arithmetic at all, so a float
//! kernel compiles to what it always did. A level the launch binds is written with
//! [`Tile::scaled`], once per level and named: binding none of them is scaling by one.
//!
//! A level above the first is read once per leaf region at its origin, so it is only correct
//! where it covers the whole tile the level below spans.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, CoordsDyn},
};

use crate::ops::matmul::leaf::{Side, check_scales_omit_rather_than_divide, check_scales_ride};
use crate::*;

/// A factor and the scales it carries, which are a pair and not a product: the values, and the
/// levels that will multiply them when the leaf reads a line. [`Tile::scaled`], said once per
/// level.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Scaled<E: Numeric, S: Numeric> {
    values: Tile<E>,
    levels: Sequence<Tile<S>>,
}

#[cube]
impl<E: Numeric> Tile<E> {
    /// This tile as a factor carrying no scales: what a float kernel contracts, and what
    /// [`mm`](Tile::mm) hands the leaf for each of its operands.
    pub fn plain(&self) -> Scaled<E, E> {
        self.unscaled::<E>()
    }

    /// [`plain`](Tile::plain) at a scale element of its own, for a factor whose levels are about
    /// to be pushed onto it.
    fn unscaled<S: Numeric>(&self) -> Scaled<E, S> {
        Scaled::<E, S> {
            values: self.clone(),
            levels: Sequence::new(),
        }
    }

    /// This factor, carrying `level`, which a scheme may not have: absent, it folds
    /// nothing and emits nothing, which is scaling by one.
    ///
    /// A scale that does not cover everything an accumulator sums rides its factor like this: the
    /// running sum already holds terms it does not apply to. One that does belongs on the
    /// accumulator ([`Tile::scale`]), one multiply per cell rather than one per value read.
    pub fn scaled<S: Numeric>(&self, level: &ComptimeOption<Tile<S>>) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        #[comptime]
        match level {
            ComptimeOption::Some(level) => levels.push(level.clone()),
            ComptimeOption::None => {}
        }
        Scaled::<E, S> {
            values: self.clone(),
            levels,
        }
    }

    /// [`scaled`](Tile::scaled) by a level the caller has in hand.
    ///
    /// `scaled` takes what a *scheme* may or may not have bound; this takes what a kernel is
    /// holding, which is the common case wherever the scales were staged rather than looked up.
    pub fn scaled_by<S: Numeric>(&self, level: Tile<S>) -> Scaled<E, S> {
        self.scaled(&ComptimeOption::new_Some(level))
    }
}

#[cube]
impl<E: Numeric, S: Numeric> Scaled<E, S> {
    /// This factor, carrying one level more, which a scheme may not have.
    pub fn scaled(&self, level: &ComptimeOption<Tile<S>>) -> Scaled<E, S> {
        let mut levels = self.levels.clone();
        #[comptime]
        match level {
            ComptimeOption::Some(level) => levels.push(level.clone()),
            ComptimeOption::None => {}
        }
        Scaled::<E, S> {
            values: self.values.clone(),
            levels,
        }
    }

    /// This factor windowed to `region`: the values and every level at once, each resolving the
    /// same position to its own granularity.
    pub fn at(&self, region: &Region) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        #[unroll]
        for i in 0..self.levels.len() {
            levels.push(self.levels.index(i).at(region));
        }
        Scaled::<E, S> {
            values: self.values.at(region),
            levels,
        }
    }

    /// The values this factor contributes.
    pub(crate) fn values(&self) -> Tile<E> {
        self.values.clone()
    }

    /// Every scale that multiplies them, innermost first. Empty is a factor carrying none.
    pub(crate) fn levels(&self) -> Sequence<Tile<S>> {
        self.levels.clone()
    }

    /// This factor's scales as a leaf reads them, looked up at the coordinates of the values'
    /// lines: `axes` is the matrix the leaf reads the values as and `matrix` which batch matrix;
    /// `side`, `out` and `acc_axes` are what the scales' statement is checked against.
    ///
    /// Every coarser level is read here, once, at the region's origin, which is once per region
    /// rather than once per value: a coarser level covers the whole region, so it has no
    /// position of its own inside it. That is the whole reason depth is cheap.
    pub(crate) fn lookup(
        &self,
        #[comptime] axes: MatrixAxes,
        matrix: usize,
        #[comptime] side: Side,
        #[comptime] out: Space,
        #[comptime] acc_axes: MatrixAxes,
    ) -> ScaleLookup<S> {
        let values = self.values();
        let vector_size = values.vector_size();
        let count = self.levels.len();
        let mut coarser = Vector::<S, Const<1>>::cast_from(1);
        #[unroll]
        for k in 1..count {
            let above = self.levels.index(k);
            let rank = comptime!(above.place.space.rank());
            let above_width = above.vector_size();
            let size!(AW) = above_width;
            let mut origin = CoordsDyn::new();
            #[unroll]
            for _axis in 0..rank {
                origin.push(0u32.runtime());
            }
            // Read as the word it lies in, and its own scale is the word's first field.
            let one = above
                .nd_packed::<AW>(comptime!(Guard::Checked))
                .read(origin)
                .extract(0usize);
            coarser *= Vector::<S, Const<1>>::cast_from(one);
        }
        let inner = if comptime!(count > 0) {
            let inner = self.levels.index(0);
            let projection = inner.projection();
            comptime!(check_scales_omit_rather_than_divide(&projection));
            comptime!(check_scales_ride(side, &inner.place.space, &out, acc_axes));
            // A line of values is under one scale: the axis it runs along is one the scales
            // omit, or the line is one value.
            let innermost = comptime!(values.place.space.axis_at(values.place.space.rank() - 1));
            comptime!(assert!(
                vector_size == 1 || !projection.addresses(innermost),
                "mm_scaled: a line runs {vector_size} values along {innermost:?}, which its \
                 scales address, so one line lies under several scales; serve the factor one \
                 value a line, or omit {innermost:?} from the scales"
            ));
            comptime!(assert!(
                inner
                    .place
                    .space
                    .axes()
                    .all(|axis| values.place.space.contains(axis)),
                "mm_scaled: the scales span {:?} where the values span {:?}; a scale is looked \
                 up at the value's coordinates, so every axis of the scales is one of the values'",
                inner.place.space.axes().collect::<Vec<_>>(),
                values.place.space.axes().collect::<Vec<_>>()
            ));
            ComptimeOption::new_Some(inner.clone())
        } else {
            ComptimeOption::new_None()
        };
        ScaleLookup::<S> {
            inner,
            coarser,
            values: comptime!(values.place.space.clone()),
            axes,
            vector_size,
            matrix,
        }
    }
}

/// A factor's scales as a leaf reads them: looked up at the coordinates of the value they
/// cover.
///
/// A scale is a tile in the values' space with the axes one scale holds whole omitted, so the
/// value's own coordinate names its scale. The nearest level is read as lines, the scale being the
/// field the coordinate falls in; every coarser level was read once at the region's origin.
///
/// Absent is a factor with no scales, and it emits nothing: its values go through as they lie.
#[derive(CubeType)]
pub struct ScaleLookup<S: Numeric> {
    /// The level nearest the values, where the factor carries any.
    inner: ComptimeOption<Tile<S>>,
    /// Every coarser level, already met and carried as one value.
    coarser: Vector<S, Const<1>>,
    /// The values' space, and the matrix the leaf reads them as: what a line's `(row, col)`
    /// resolves to a coordinate through.
    #[cube(comptime)]
    values: Space,
    #[cube(comptime)]
    axes: MatrixAxes,
    #[cube(comptime)]
    vector_size: usize,
    /// Which batch matrix the lines are read from.
    matrix: usize,
}

#[cube]
impl<S: Numeric> ScaleLookup<S> {
    /// Whether a read reaches the lane that asks by a plane shuffle ([`Tile::by_shuffle`]),
    /// so the reader keeps its lanes converged around it.
    pub fn by_shuffle(&self) -> comptime_type!(bool) {
        #[comptime]
        match &self.inner {
            ComptimeOption::Some(inner) => inner.by_shuffle(),
            ComptimeOption::None => comptime!(false),
        }
    }

    /// `value`, the line at `pos` of the values' matrix, under the scale covering it.
    pub fn apply<E: Numeric, V: Size>(&self, value: Vector<E, V>, pos: Coords2d) -> Vector<E, V> {
        let (row, col) = pos;
        let coords = matrix_coords(
            row,
            col,
            self.matrix,
            comptime!(&self.values),
            comptime!(self.axes),
            comptime!(self.vector_size),
        );
        self.apply_at::<E, V>(value, &coords)
    }

    /// `value`, the line whose first value lies at `coords` of the values' space, under the
    /// scale covering it.
    pub fn apply_at<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        coords: &Coords<u32>,
    ) -> Vector<E, V> {
        #[comptime]
        match &self.inner {
            ComptimeOption::Some(inner) => {
                let scale = inner.scale_for(coords, comptime!(self.values.clone()));
                value * Vector::<E, V>::cast_from(scale * self.coarser.extract(0usize))
            }
            ComptimeOption::None => value,
        }
    }
}

#[cube]
impl<S: Numeric> Tile<S> {
    /// Whether a read of this tile reaches the lane that asks by a plane shuffle, which the
    /// whole plane takes part in: a reader must keep its lanes converged around it. True of the
    /// plane's own lanes ([`Lines`]) and of nothing else.
    pub(crate) fn by_shuffle(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Lines(_) => comptime!(true),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => comptime!(false),
        }
    }

    /// The one scale at `coords`, one entry per axis of this tile's space, through whatever
    /// holds it: the plane's lanes are read at the coordinate itself; a memory tile serves
    /// lines, so the coordinate names a line and the field of it the scale sits in.
    pub(crate) fn scale_at(&self, coords: &Coords<u32>) -> S {
        match &self.kind {
            TileKind::Lines(lines) => lines.read(coords),
            TileKind::Memory(_) => self.value_in_line(coords),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("Tile::scale_at: a scale is read from memory or from the plane's lanes")
            }
        }
    }

    /// The one scale covering the value at `coords` of a tile spanning `values`.
    ///
    /// A scale's space omits the axes one scale holds whole, so the value's own coordinate along
    /// each axis the scales do carry names the scale, and the omitted ones contribute nothing.
    pub(crate) fn scale_for(&self, coords: &Coords<u32>, #[comptime] values: Space) -> S {
        let rank = comptime!(self.place.space.rank());
        let mut own = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let axis = comptime!(self.place.space.axis_at(p));
            own.push(coords.at(comptime!(values.position(axis))));
        }
        self.scale_at(&own)
    }

    /// The one value at `coords` of a tile that serves lines: every coordinate but the innermost
    /// names the line outright, and the innermost splits into the line and the field of it the
    /// value sits in.
    fn value_in_line(&self, coords: &Coords<u32>) -> S {
        let rank = coords.len();
        let width = self.vector_size();
        let size!(W) = width;
        let mut at = CoordsDyn::new();
        let mut field = 0u32.runtime();
        #[unroll]
        for p in 0..rank {
            let coord = coords.at(p);
            if comptime!(p == rank - 1 && width > 1) {
                field = coord.remainder(comptime!(width as u32));
                at.push(coord.divided_by(comptime!(width as u32)));
            } else {
                at.push(coord);
            }
        }
        let line = self.nd_packed::<W>(comptime!(Guard::Checked)).read(at);
        if comptime!(width > 1) {
            line.extract_dynamic(field.retyped::<usize>())
        } else {
            line.extract(0usize)
        }
    }
}

/// A scale level a launch may or may not have bound, as the tile it serves: `u32` words read in
/// the level's own width, which its [`Field`](crate::Field) states and nothing here asks about.
#[cube]
pub fn scale_tile<S: Numeric>(
    level: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
    #[comptime] space: Partitioning,
) -> ComptimeOption<Tile<S>> {
    #[comptime]
    match level {
        ComptimeOption::Some(level) => {
            ComptimeOption::new_Some(level.tile_as::<S>(comptime!(space.clone())))
        }
        ComptimeOption::None => ComptimeOption::new_None(),
    }
}

/// A scale level windowed the way the factor it multiplies is, or nothing where none was bound.
#[cube]
pub trait MaybeTile: CubeType {
    /// The element the level is served at.
    type E: Numeric;

    /// This level at `region`, descending it as [`Tile::at`] descends a factor.
    fn at(&self, region: &Region) -> ComptimeOption<Tile<Self::E>>;

    /// This level as the steps under one region of `level` read it: a stage refilled once a
    /// region with [`copy_from`](MaybeTile::copy_from). Where `level` names none the steps read
    /// the level where it lies, and this is that level.
    ///
    /// A level a scheme never bound stages nothing and stays absent, so a kernel stages its
    /// scales without asking whether it has any.
    fn staged(
        &self,
        #[comptime] level: Option<Level>,
        #[comptime] storage: StageStorage,
    ) -> ComptimeOption<Tile<Self::E>>;

    /// This level filled from `src`, where both are there; nothing where either is not.
    ///
    /// The pair is what a stage and the scales it stages are: they are bound together or not at
    /// all, so a caller says "fill" once rather than testing both.
    fn copy_from(&mut self, src: &ComptimeOption<Tile<Self::E>>);
}

#[cube]
impl<E: Numeric> MaybeTile for ComptimeOption<Tile<E>> {
    type E = E;

    fn at(&self, region: &Region) -> ComptimeOption<Tile<E>> {
        #[comptime]
        match self {
            ComptimeOption::Some(level) => ComptimeOption::new_Some(level.at(region)),
            ComptimeOption::None => ComptimeOption::new_None(),
        }
    }

    fn staged(
        &self,
        #[comptime] level: Option<Level>,
        #[comptime] storage: StageStorage,
    ) -> ComptimeOption<Tile<E>> {
        match comptime!(level) {
            None => self.clone(),
            Some(level) =>
            {
                #[comptime]
                match self {
                    ComptimeOption::Some(scales) => ComptimeOption::new_Some(Memory::<E>::stage(
                        scales,
                        comptime!(level.clone()),
                        comptime!(storage.clone()),
                        comptime!(None),
                    )),
                    ComptimeOption::None => ComptimeOption::new_None(),
                }
            }
        }
    }

    fn copy_from(&mut self, src: &ComptimeOption<Tile<E>>) {
        #[comptime]
        if let (ComptimeOption::Some(stage), ComptimeOption::Some(src)) = (self, src) {
            stage.copy_from(src);
        }
    }
}
