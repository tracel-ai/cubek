//! The chunk: the stage a plane holds for the steps it walks under one load of its scales.
//!
//! Lane `t` holds line `t`, loaded once for the chunk by one coalesced read of the plane. A
//! value's scale is then read at the value's coordinates — the line named by the coordinates
//! along the lines' axes, the byte by the coordinate along the line — and reaches the lane that
//! asks by a plane shuffle, a word at a time. Where the plane cannot shuffle, the same lines lie
//! in the plane's own shared window instead, written once a chunk and read by index.
//!
//! The lines are held as the words they lie in and decoded at the read: four `ue4m3` bytes to a
//! word, or a whole `f32`. Nothing here windows a line. A window into a chunk is a scalar origin,
//! so a step one block deep into a line of four is a coordinate, never a line index a window
//! cannot state ([`MemData::at`](crate::MemData) refuses exactly that cut).

use std::marker::PhantomData;

use cubecl::{ir::ElemType, prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

// Words one line holds, as a scope-registered size rather than a generic: bound where the chunk
// is opened, so the width stays a storage detail of the chunk and never reaches `TileKind`.
define_size!(pub(crate) LW);

/// What a coordinate along each axis of `chunk` counts in lines: row-major over the axes
/// `projection` addresses above the line, zero along the rest.
fn line_strides(chunk: &Space, projection: &Projection) -> Vec<usize> {
    let rank = chunk.rank();
    let mut strides = vec![0; rank];
    let mut stride = 1;
    for p in (0..rank - 1).rev() {
        if projection.addresses(chunk.axis_at(p)) {
            strides[p] = stride;
            stride *= chunk.extent_at(p);
        }
    }
    strides
}

/// Lines `chunk` holds: one per position along the axes `projection` addresses above the line.
fn lines_of(chunk: &Space, projection: &Projection) -> usize {
    (0..chunk.rank() - 1)
        .filter(|&p| projection.addresses(chunk.axis_at(p)))
        .map(|p| chunk.extent_at(p))
        .product()
}

/// Bind the line width `LW` for the rest of the kernel's scope.
#[cube]
fn register_line_words(#[comptime] words: usize) {
    intrinsic!(|scope| {
        scope.register_size::<LW>(words);
    });
}

/// The lines a plane holds for one chunk of its walk, and where inside them a window sits.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Chunk<T: Numeric> {
    /// This lane's line, as the words it lies in, where the plane shuffles: lane `t` holds
    /// line `t`. One entry, held in an array so a load can replace it.
    lanes: Array<Vector<u32, LW>>,
    /// Every line, one after the other, where the plane's shared window holds them instead.
    landing: ComptimeOption<Shared<[u32]>>,
    /// Where this window starts inside the chunk, in scalars, one entry per axis of the chunk.
    origin: Coords<u32>,
    /// The box the chunk was loaded as: the lines' axes, then the line's.
    #[cube(comptime)]
    chunk: Space,
    /// What a coordinate along each axis of the box counts in lines: the product of the
    /// addressed extents inside it, and nothing along an axis the operand spans without
    /// addressing (one scale holding the whole of it) or along the line itself.
    #[cube(comptime)]
    strides: Vec<usize>,
    /// The projection of the operand this chunk stages: which of its axes address a line and
    /// which one scale holds whole.
    #[cube(comptime)]
    projection: Projection,
    /// The slot one value occupies in a word.
    #[cube(comptime)]
    field: Field,
    /// Words one line holds.
    #[cube(comptime)]
    words: usize,
    /// Whether the lines are held in the lanes and read by shuffle.
    #[cube(comptime)]
    broadcast: bool,
    #[cube(comptime)]
    _served: PhantomData<T>,
}

#[cube]
impl<T: Numeric> Chunk<T> {
    /// The chunk one region of `level` over `operand` fills, held in the lanes (`broadcast`) or
    /// in the plane's shared window, empty until [`load`](Self::load).
    pub(crate) fn new(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] broadcast: bool,
    ) -> Chunk<T> {
        let chunk = comptime!(level.child(&operand.space));
        let rank = comptime!(chunk.rank());
        let line = comptime!(chunk.extent_at(rank - 1));
        let projection = operand.projection();
        comptime!(assert!(
            projection.addresses(chunk.axis_at(rank - 1)),
            "Chunk: the line runs along the operand's innermost axis, which it must address"
        ));
        let strides = comptime!(line_strides(&chunk, &projection));
        let lines = comptime!(lines_of(&chunk, &projection));
        let packing = operand.packing();
        let served = elem_type_of::<T>();
        // The lines are held as words: a packed operand's as they lie, a plain one's as the
        // 32-bit values it serves, reinterpreted.
        let field = comptime!(match packing {
            Packing::Packed { field } => field,
            Packing::Plain => match served {
                ElemType::Float(kind) if float_field_bits(kind) == 32 => Field::Float(kind),
                other => panic!(
                    "Chunk: a plain operand is held as whole 32-bit words, and {other:?} is not \
                     one; bind the scales packed, or serve them as `f32`"
                ),
            },
            Packing::Native => panic!("Chunk: a native store has no words to hold"),
        });
        let per_word = comptime!(field.per_word());
        comptime!(assert!(
            line.is_multiple_of(per_word),
            "Chunk: a line of {line} values is not whole words of {per_word}"
        ));
        let words = comptime!(line / per_word);
        register_line_words(words);
        let landing = if comptime!(broadcast) {
            ComptimeOption::new_None()
        } else {
            // One window per plane of the cube, this plane's found by the walk's own decode of
            // the hardware position, as a landing is.
            let planes = comptime!(plane_windows(&operand.space, &operand.levels));
            let cells = comptime!(lines * words);
            let start = hardware_pos(ComputeScope::Plane) * cells;
            let end = start + cells;
            ComptimeOption::new_Some(
                Shared::<[u32]>::new_slice(comptime!(cells * planes)).map(|all| &all[start..end]),
            )
        };
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for _axis in 0..rank {
            origin.push(0u32.runtime());
        }
        Chunk::<T> {
            lanes: Array::<Vector<u32, LW>>::new(1usize),
            landing,
            origin,
            chunk,
            strides,
            projection,
            field,
            words,
            broadcast,
            _served: PhantomData,
        }
    }

    /// Lines the chunk holds.
    pub(crate) fn lines(&self) -> comptime_type!(usize) {
        let rank = comptime!(self.chunk.rank());
        comptime!(
            (0..rank - 1)
                .filter(|&p| self.strides[p] > 0)
                .map(|p| self.chunk.extent_at(p))
                .product::<usize>()
        )
    }

    /// Values one line holds: what a read of the operand this chunk stages serves.
    pub(crate) fn line(&self) -> comptime_type!(usize) {
        let rank = comptime!(self.chunk.rank());
        comptime!(self.chunk.extent_at(rank - 1))
    }

    /// Whether a read reaches the lane that asks by a plane shuffle, which the whole plane
    /// takes part in: a reader must keep its lanes converged around it.
    pub(crate) fn by_shuffle(&self) -> comptime_type!(bool) {
        comptime!(self.broadcast)
    }

    /// The projection of the operand this chunk stages.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        comptime!(self.projection.clone())
    }

    /// How the operand this chunk stages is packed, as the chunk holds it.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(Packing::Packed { field: self.field })
    }

    /// Load the chunk from `src`, the memory window of the box it was opened over: lane `t`
    /// reads line `t`, whole — in one read where the source serves a line at a time, else in
    /// the few consecutive reads a line takes. Lanes past the lines read nothing.
    pub(crate) fn load(&mut self, src: &Tile<T>) {
        let rank = comptime!(self.chunk.rank());
        let line = self.line();
        let lines = self.lines();
        let served = src.vector_size();
        comptime!(assert!(
            line.is_multiple_of(served),
            "Chunk::copy_from: a lane reads its line of {line} values in whole reads, and the \
             source serves {served} a read"
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
                // The line's coordinate along every axis: its digits along the addressed ones,
                // the origin along the rest (which one line holds whole), and along the line
                // itself the read, counted in the lines the source serves.
                let mut at = CoordsDyn::new();
                #[unroll]
                for p in 0..rank - 1 {
                    let stride = comptime!(self.strides[p]);
                    if comptime!(stride > 0) {
                        at.push(
                            lane.fdiv(comptime!(stride as u32))
                                .frem(comptime!(self.chunk.extent_at(p) as u32)),
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
                    Packing::Native => panic!("Chunk::copy_from: a native store has no words"),
                };
                #[unroll]
                for j in 0..per_read {
                    held.insert(comptime!(r * per_read + j), got.extract(j));
                }
            }
        }
        if comptime!(self.broadcast) {
            self.lanes[0usize] = held;
        } else {
            #[comptime]
            match &mut self.landing {
                ComptimeOption::Some(landing) => {
                    if lane < comptime!(lines as u32) {
                        let base = (lane * comptime!(words as u32)) as usize;
                        #[unroll]
                        for j in 0..words {
                            landing[base + j] = held.extract(j);
                        }
                    }
                    sync_plane();
                }
                ComptimeOption::None => {}
            }
        }
    }

    /// This chunk windowed one level down, to `step`'s box: the origin moves, in scalars, and
    /// nothing is cropped.
    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> Chunk<T> {
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
        Chunk::<T> {
            lanes: self.lanes,
            landing: self.landing.clone(),
            origin,
            chunk: comptime!(self.chunk.clone()),
            strides: comptime!(self.strides.clone()),
            projection: comptime!(self.projection.clone()),
            field: comptime!(self.field),
            words: comptime!(self.words),
            broadcast: comptime!(self.broadcast),
            _served: PhantomData,
        }
    }

    /// The value at `coords` of this window, one entry per axis of the chunk: the line the
    /// coordinates along the lines' axes name, the word and the field of it the coordinate
    /// along the line names.
    pub(crate) fn read(&self, coords: &Coords<u32>) -> T {
        let rank = comptime!(self.chunk.rank());
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
        let held = if comptime!(self.broadcast) {
            // Every lane offers its word `j`; the lane that asked receives line `line`'s.
            let mine = self.lanes[0usize];
            let mut got = Vector::<u32, LW>::empty();
            #[unroll]
            for j in 0..words {
                got.insert(j, plane_shuffle(mine.extract(j), line));
            }
            if comptime!(words > 1) {
                got.extract_dynamic(word.fcast::<usize>())
            } else {
                got.extract(0usize)
            }
        } else {
            #[comptime]
            match &self.landing {
                ComptimeOption::Some(landing) => {
                    landing[(line.fmul(comptime!(words as u32)).fadd(word)) as usize]
                }
                ComptimeOption::None => panic!("Chunk: no lines are held"),
            }
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
