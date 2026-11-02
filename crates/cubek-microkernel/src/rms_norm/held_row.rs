use crate::row_store::OwnerLines;
use cubecl::ir::ElemType;
use cubecl::tile::{Monoid, Tile};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::{ArithmeticInstruction, UnitWork};

/// How an owner's units share the lines of the row they hold: `lines` lines of the tile, each of
/// `units` units taking every `units`-th, so consecutive units read consecutive lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeldRowShape {
    /// Lines of the tile the owner holds: its run of the row, in vectors of the operands' width.
    pub lines: usize,
    /// Units sharing them.
    pub units: usize,
}

impl HeldRowShape {
    /// The most lines one unit holds: past it the registers, not the row, bound the kernel, and an
    /// unrolled walk of its share takes longer to compile than the launch it serves takes to run.
    pub const MAX_LINES_PER_UNIT: usize = 16;

    /// Lines one unit holds: every `units`-th of the run, the last unit's share one short where
    /// the units do not divide it.
    pub fn per_unit(&self) -> usize {
        self.lines.div_ceil(self.units.max(1))
    }

    /// What one unit holds of the work, its lines `width` elements of `element` wide: the lines in
    /// registers beside its partial, a lower bound counting what it keeps live and not what the
    /// compiler adds, squared with the device's vector arithmetic.
    pub fn work(&self, width: usize, element: ElemType) -> UnitWork {
        let bytes = self.per_unit() * width * element.size();
        UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: bytes.div_ceil(4) as u32 + 1,
        }
    }
}

/// One unit's share of an owner's row, in registers between the row's two passes: read once,
/// squared into the unit's partial in `f32` as it lands — an `f16` square-sum overflows at a
/// magnitude of about 256 — then scaled and written out of the registers.
///
/// Unit `unit` holds lines `unit + i · units` of the tile it is handed ([`HeldRowShape`]).
#[derive(CubeType)]
pub(crate) struct HeldRow<E: Float, N: Size> {
    lines: Array<Vector<E, N>>,
    #[cube(comptime)]
    shape: HeldRowShape,
}

#[cube]
impl<E: Float, N: Size> HeldRow<E, N> {
    /// Registers for one unit's share of a row laid out as `shape` says, holding nothing yet.
    pub fn new(#[comptime] shape: HeldRowShape) -> Self {
        HeldRow::<E, N> {
            lines: Array::new(comptime!(shape.per_unit())),
            shape,
        }
    }

    /// Unit `unit`'s lines of `input`, windowed to the owner's run, read into the registers;
    /// returns the sum of their squares.
    ///
    /// The view masks the tensor's bounds, not the run's, so a unit's last line, which lies past
    /// the run where the units do not divide it, is read clamped onto the run's last and left out
    /// of the sum. It is read rather than skipped so that every register holds a line: one left at
    /// the zero the compiler starts it at is a zero vector cubecl's constant evaluation cannot cast.
    pub fn load(&mut self, input: &Tile, unit: usize) -> f32 {
        let shape = comptime!(self.shape);
        let width = input.vector_size();
        let read = input.flat::<E, N>();
        let mut partial = 0.0f32;
        #[unroll]
        for i in 0..comptime!(shape.per_unit()) {
            let line = unit + i * comptime!(shape.units);
            let held = read.read(min(line, comptime!(shape.lines - 1)));
            self.lines[i] = held;
            if comptime!(shape.lines.is_multiple_of(shape.units)) || line < comptime!(shape.lines) {
                let wide = Vector::<f32, N>::cast_from(held);
                partial += Monoid::reduce::<f32, N>(wide * wide, width, Monoid::Sum);
            }
        }
        partial
    }

    /// Unit `unit`'s lines written into `out`, each scaled by `factor` in `f32`, cast back, and
    /// multiplied by its line of `weight`; `out` and `weight` are the owner's run, and a line past
    /// it is not written.
    pub fn write(&self, weight: &Tile, out: &mut OwnerLines, unit: usize, factor: f32) {
        let shape = comptime!(self.shape);
        let weights = weight.flat::<E, N>();
        let factor = Vector::<f32, N>::cast_from(factor);
        #[unroll]
        for i in 0..comptime!(shape.per_unit()) {
            let line = unit + i * comptime!(shape.units);
            if comptime!(shape.lines.is_multiple_of(shape.units)) || line < comptime!(shape.lines) {
                let normalized =
                    Vector::<E, N>::cast_from(Vector::<f32, N>::cast_from(self.lines[i]) * factor);
                out.write::<E, N>(line, normalized * weights.read(line));
            }
        }
    }
}
