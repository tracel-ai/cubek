//! A tile the plane holds in its lanes: loaded once, coalesced, and shared by shuffle.
//!
//! Lane `t` holds line `t`. A value is reached at its coordinates, the line named by the
//! coordinates along the lines' axes and the byte by the coordinate along the line, and arrives
//! at the lane that asks by a plane shuffle, a word at a time.
//!
//! **One line per lane**, so a box deeper than the plane is wide does not fit. The plane's width
//! is the launch's rather than a comptime fact, so nothing here can refuse one; a caller wanting
//! more lines than the device gives it lanes wants a shared-memory stage, which is a different
//! tile and not a mode of this one.
//!
//! The lines are held as the words they lie in and decoded at the read. Nothing here windows a
//! line: a window into the box is a scalar origin, so a step one block deep into a line of four
//! is a coordinate, never a line index a window cannot state
//! ([`MemData::at`](crate::MemData) refuses exactly that cut).

use std::marker::PhantomData;

use cubecl::{ir::ElemType, prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

// Words one line holds, as a scope-registered size rather than a generic: bound where the tile
// is opened, so the width stays a storage detail of it and never reaches `TileKind`.
define_size!(pub(crate) LW);

/// Bind the line width `LW` for the rest of the kernel's scope.
#[cube]
fn register_line_words(#[comptime] words: usize) {
    intrinsic!(|scope| {
        scope.register_size::<LW>(words);
    });
}

/// The lines a plane holds in its lanes, and where inside them a window sits.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Lanes<T: Numeric> {
    /// This lane's line, as the words it lies in: lane `t` holds line `t`. One entry, held in an
    /// array so a load can replace it.
    lanes: Array<Vector<u32, LW>>,
    /// Where this window starts inside the box, in scalars, one entry per axis of it.
    origin: Coords<u32>,
    /// The box the lines were loaded as: the lines' axes, then the line's. The tile's own space
    /// narrows as [`at`](Lanes::at) windows it; this stays what was loaded.
    #[cube(comptime)]
    loaded: Space,
    /// What a coordinate along each axis of the box counts in lines: the product of the
    /// addressed extents inside it, and nothing along an axis the operand spans without
    /// addressing (one value holding the whole of it) or along the line itself.
    #[cube(comptime)]
    strides: Vec<usize>,
    /// Lines the box holds, which is the lanes it takes.
    #[cube(comptime)]
    lines: usize,
    /// The projection of the operand these lines stage: which of its axes address a line and
    /// which one value holds whole.
    #[cube(comptime)]
    projection: Projection,
    /// The slot one value occupies in a word.
    #[cube(comptime)]
    field: Field,
    /// Words one line holds.
    #[cube(comptime)]
    words: usize,
    #[cube(comptime)]
    _served: PhantomData<T>,
}

impl<T: Numeric> Lanes<T> {
    /// How `loaded` counts in lines: what a coordinate along each of its axes is worth, row-major
    /// over the axes `projection` addresses above the line and zero along the rest, and how many
    /// lines that leaves. The count is the last stride the walk writes, so one pass gives both.
    fn line_strides(loaded: &Space, projection: &Projection) -> (Vec<usize>, usize) {
        let rank = loaded.rank();
        let mut strides = vec![0; rank];
        let mut lines = 1;
        // Indexed, not iterated: each addressed stride is the product of the ones written below it.
        #[allow(clippy::needless_range_loop)]
        for p in (0..rank - 1).rev() {
            if projection.addresses(loaded.axis_at(p)) {
                strides[p] = lines;
                lines *= loaded.extent_at(p);
            }
        }
        (strides, lines)
    }
}

#[cube]
impl<T: Numeric> Lanes<T> {
    /// The box one region of `level` over `operand` fills, held one line to a lane, empty until
    /// [`load`](Self::load).
    pub(crate) fn new(operand: &Tile<T>, #[comptime] level: Level) -> Lanes<T> {
        let loaded = comptime!(level.child(&operand.space));
        let rank = comptime!(loaded.rank());
        let line = comptime!(loaded.extent_at(rank - 1));
        let projection = operand.projection();
        comptime!(assert!(
            projection.addresses(loaded.axis_at(rank - 1)),
            "Lanes: the line runs along the operand's innermost axis, which it must address"
        ));
        let (strides, lines) = comptime!(Lanes::<T>::line_strides(&loaded, &projection));
        let packing = operand.packing();
        let served = elem_type_of::<T>();
        // The lines are held as words: a packed operand's as they lie, a plain one's as the
        // 32-bit values it serves, reinterpreted.
        let field = comptime!(match packing {
            Packing::Packed { field } => field,
            Packing::Plain => match served {
                ElemType::Float(kind) if float_field_bits(kind) == 32 => Field::Float(kind),
                other => panic!(
                    "Lanes: a plain operand is held as whole 32-bit words, and {other:?} is not \
                     one; bind the operand packed, or serve it as `f32`"
                ),
            },
            Packing::Native => panic!("Lanes: a native store has no words to hold"),
        });
        let per_word = comptime!(field.per_word());
        comptime!(assert!(
            line.is_multiple_of(per_word),
            "Lanes: a line of {line} values is not whole words of {per_word}"
        ));
        let words = comptime!(line / per_word);
        register_line_words(words);
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for _axis in 0..rank {
            origin.push(0u32.runtime());
        }
        Lanes::<T> {
            lanes: Array::<Vector<u32, LW>>::new(1usize),
            origin,
            loaded,
            strides,
            lines,
            projection,
            field,
            words,
            _served: PhantomData,
        }
    }

    /// Lines the box holds.
    pub(crate) fn lines(&self) -> comptime_type!(usize) {
        comptime!(self.lines)
    }

    /// Values one line holds: what a read of the operand these lines stage serves.
    pub(crate) fn line(&self) -> comptime_type!(usize) {
        let rank = comptime!(self.loaded.rank());
        comptime!(self.loaded.extent_at(rank - 1))
    }

    /// The projection of the operand these lines stage.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        comptime!(self.projection.clone())
    }

    /// How the operand these lines stage is packed, as they hold it.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(Packing::Packed { field: self.field })
    }

    /// Load from `src`, the memory window of the box: lane `t` reads line `t`, whole — in one
    /// read where the source serves a line at a time, else in the few consecutive reads a line
    /// takes. Lanes past the lines read nothing.
    pub(crate) fn load(&mut self, src: &Tile<T>) {
        let rank = comptime!(self.loaded.rank());
        let line = self.line();
        let lines = self.lines();
        let served = src.vector_size();
        comptime!(assert!(
            line.is_multiple_of(served),
            "Lanes::load: a lane reads its line of {line} values in whole reads, and the source \
             serves {served} a read"
        ));
        let reads = comptime!(line / served);
        let words = comptime!(self.words);
        let per_read = comptime!(words / reads);
        let size!(WR) = per_read;
        let packing = src.packing();
        let lane = UNIT_POS_PLANE;
        let mut held = Vector::<u32, LW>::empty();
        if lane < comptime!(lines as u32) {
            #[unroll]
            for r in 0..reads {
                // The line's coordinate along every axis: its digits along the addressed
                // ones, zero along the rest (which one line holds whole), and the read along the
                // line itself, counted in the lines the source serves.
                let mut at = CoordsDyn::new();
                #[unroll]
                for p in 0..rank - 1 {
                    let stride = comptime!(self.strides[p]);
                    if comptime!(stride > 0) {
                        at.push(
                            lane.fdiv(comptime!(stride as u32))
                                .frem(comptime!(self.loaded.extent_at(p) as u32)),
                        );
                    } else {
                        at.push(0u32.runtime());
                    }
                }
                at.push(comptime!(r as u32).runtime());
                let got = match comptime!(packing) {
                    Packing::Packed { field: _ } => {
                        src.nd_words::<WR>(comptime!(Guard::Checked)).read(at)
                    }
                    Packing::Plain => {
                        let size!(SV) = served;
                        let values = src.nd_packed::<SV>(comptime!(Guard::Checked)).read(at);
                        let mut bits = Vector::<u32, WR>::empty();
                        #[unroll]
                        for j in 0..per_read {
                            bits.insert(j, u32::reinterpret(values.extract(j)));
                        }
                        bits
                    }
                    Packing::Native => panic!("Lanes::load: a native store has no words"),
                };
                #[unroll]
                for j in 0..per_read {
                    held.insert(comptime!(r * per_read + j), got.extract(j));
                }
            }
        }
        self.lanes[0usize] = held;
    }

    /// This window one level down, to `step`'s box: the origin moves, in scalars, and nothing is
    /// cropped.
    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> Lanes<T> {
        let rank = comptime!(space.rank());
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let axis = comptime!(space.axis_at(p));
            let edge = comptime!(step.level.extent_in(&space, axis).get());
            origin.push(
                self.origin
                    .at(p)
                    .fadd(step.coord(axis).fmul(edge).fcast::<u32>()),
            );
        }
        Lanes::<T> {
            lanes: self.lanes,
            origin,
            loaded: comptime!(self.loaded.clone()),
            strides: comptime!(self.strides.clone()),
            lines: comptime!(self.lines),
            projection: comptime!(self.projection.clone()),
            field: comptime!(self.field),
            words: comptime!(self.words),
            _served: PhantomData,
        }
    }

    /// The value at `coords` of this window, one entry per axis of the box: the line the
    /// coordinates along the lines' axes name, the word and the field of it the coordinate
    /// along the line names.
    ///
    /// The shuffle is the whole plane's, so a caller keeps its lanes converged around this.
    pub(crate) fn read(&self, coords: &Coords<u32>) -> T {
        let rank = comptime!(self.loaded.rank());
        let mut line = 0u32.runtime();
        #[unroll]
        for p in 0..rank - 1 {
            let stride = comptime!(self.strides[p] as u32);
            if comptime!(stride > 0) {
                line = line.fadd(self.origin.at(p).fadd(coords.at(p)).fmul(stride));
            }
        }
        let byte = self.origin.at(rank - 1).fadd(coords.at(rank - 1));
        let per_word = comptime!(self.field.per_word());
        let words = comptime!(self.words);
        let word = byte.fdiv(comptime!(per_word as u32));
        let field = byte.frem(comptime!(per_word as u32));
        // Every lane offers its word `j`; the lane that asked receives line `line`'s.
        let mine = self.lanes[0usize];
        let mut got = Vector::<u32, LW>::empty();
        #[unroll]
        for j in 0..words {
            got.insert(j, plane_shuffle(mine.extract(j), line));
        }
        let held = if comptime!(words > 1) {
            got.extract_dynamic(word.fcast::<usize>())
        } else {
            got.extract(0usize)
        };
        let size!(PW) = per_word;
        let unpacked = unpack_line::<T, Const<1>, PW>(
            Vector::<u32, Const<1>>::new(held),
            comptime!(self.field),
        );
        if comptime!(per_word > 1) {
            unpacked.extract_dynamic(field.fcast::<usize>())
        } else {
            unpacked.extract(0usize)
        }
    }
}
