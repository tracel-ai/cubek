//! What a contraction reads as one factor of its terms: an operand's values, and the scales that
//! multiply them.
//!
//! A quantized tensor is values *and* scales, and an operand's binding names one thing, so the
//! scales are an operand of their own and folding them in is the contraction's own arithmetic.
//! Which factor they multiply is where the kernel wrote them. How deep they go is how many times
//! it said so. Nothing here states a side and nothing counts levels.
//!
//! **Nothing here multiplies anything.** `scaled` says which level a factor carries, and the
//! multiply happens once per line, at the read, inside the leaf ([`ScaleLookup`]) — which is
//! the only place it can happen without materializing a dequantized tile. So a factor that
//! says `scaled` twice is multiplied twice per line, not twice up front, and the verb that
//! applies them is the contraction's ([`mm_scaled`](Tile::mm_scaled) and its twins).
//!
//! A factor that carries none is [`Tile::plain`], and the leaf reads it with no arithmetic at
//! all, which is why a float kernel compiles to what it always did. A level the launch binds is
//! written with [`Tile::scaled`], once per level and named: binding none of them is
//! scaling by one.
//!
//! A level above the first is read once per leaf region at its origin, so it is only correct
//! where it covers the whole tile the level below spans.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, CoordsDyn},
};

use crate::instruction::registers::contract::{
    Side, check_scales_omit_rather_than_divide, check_scales_ride,
};
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
    /// A scale that does not cover everything an accumulator sums has to ride its factor like
    /// this, because the running sum already holds terms it does not apply to. One that does
    /// cover everything belongs on the accumulator instead ([`Tile::scale`]), where it costs one
    /// multiply per cell rather than one per value read.
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
            let rank = comptime!(above.space.rank());
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
            comptime!(check_scales_ride(side, &inner.space, &out, acc_axes));
            // A line of values is under one scale: the axis it runs along is one the scales
            // omit, or the line is one value.
            let innermost = comptime!(values.space.axis_at(values.space.rank() - 1));
            comptime!(assert!(
                vector_size == 1 || !projection.addresses(innermost),
                "mm_scaled: a line runs {vector_size} values along {innermost:?}, which its \
                 scales address, so one line lies under several scales; serve the factor one \
                 value a line, or omit {innermost:?} from the scales"
            ));
            comptime!(assert!(
                inner.space.axes().all(|axis| values.space.contains(axis)),
                "mm_scaled: the scales span {:?} where the values span {:?}; a scale is looked \
                 up at the value's coordinates, so every axis of the scales is one of the values'",
                inner.space.axes().collect::<Vec<_>>(),
                values.space.axes().collect::<Vec<_>>()
            ));
            ComptimeOption::new_Some(inner.clone())
        } else {
            ComptimeOption::new_None()
        };
        ScaleLookup::<S> {
            inner,
            coarser,
            values: comptime!(values.space.clone()),
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
/// value's own coordinate names its scale: the coordinate of every axis the scales address is
/// the value's, and the axes they omit contribute nothing. The level nearest the values is read
/// as the lines its binding serves, and the scale is the field of that line the coordinate
/// falls in — a shift and a byte where the scales come four to a word. Every coarser level was
/// read once, at the region's origin, and rides along as one value.
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
    /// `value`, the line at `pos` of the values' matrix, under the scale covering it.
    pub fn apply<E: Numeric, V: Size>(&self, value: Vector<E, V>, pos: Coords2d) -> Vector<E, V> {
        #[comptime]
        match &self.inner {
            ComptimeOption::Some(inner) => {
                let (row, col) = pos;
                let coords = matrix_coords(
                    row,
                    col,
                    self.matrix,
                    comptime!(&self.values),
                    comptime!(self.axes),
                    comptime!(self.vector_size),
                );
                let sw = inner.vector_size();
                let size!(SW) = sw;
                let (at, field) = scale_coords(
                    &coords,
                    comptime!(self.values.clone()),
                    comptime!(inner.space.clone()),
                    sw,
                );
                let line = inner.nd_packed::<SW>(comptime!(Guard::Checked)).read(at);
                let scale = if comptime!(sw > 1) {
                    line.extract_dynamic(field.fcast::<usize>())
                } else {
                    line.extract(0usize)
                };
                value * Vector::<E, V>::cast_from(scale * self.coarser.extract(0usize))
            }
            ComptimeOption::None => value,
        }
    }
}

/// Where the scale covering the value at `coords` lies: the scales' own coordinate, one entry
/// per axis of their space, its innermost a line index; and the field of that line the value's
/// coordinate falls in.
#[cube]
fn scale_coords(
    coords: &Coords<u32>,
    #[comptime] values: Space,
    #[comptime] scales: Space,
    #[comptime] sw: usize,
) -> (CoordsDyn, u32) {
    let rank = comptime!(scales.rank());
    let mut pos = CoordsDyn::new();
    let mut field = 0u32.runtime();
    #[unroll]
    for p in 0..rank {
        let axis = comptime!(scales.axis_at(p));
        let coord = coords.at(comptime!(values.position(axis)));
        if comptime!(p == rank - 1 && sw > 1) {
            field = coord.frem(comptime!(sw as u32));
            pos.push(coord.fdiv(comptime!(sw as u32)));
        } else {
            pos.push(coord);
        }
    }
    (pos, field)
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
}
