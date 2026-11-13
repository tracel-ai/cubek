use crate::activation;
use crate::rms_norm::GateOrder;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// What an owner's units read a row's lines from: the input's tile, and the gate's where the norm
/// is gated. It emits the lines the norm sums and scales, which are the input's, or the input's
/// times the silu of the gate's where the gate comes first, and applies the gate to the lines the
/// norm writes where it comes last.
#[derive(CubeType)]
pub struct RowReader {
    input: Tile,
    gate: ComptimeOption<Tile>,
    #[cube(comptime)]
    order: Option<GateOrder>,
}

#[cube]
impl RowReader {
    /// The reader of `input` and, where the norm is gated, `gate`, both windowed to the owner's
    /// run.
    pub fn new(
        input: &Tile,
        gate: &ComptimeOption<Tile>,
        #[comptime] order: Option<GateOrder>,
    ) -> Self {
        RowReader {
            input: input.clone(),
            gate: gate.clone(),
            order,
        }
    }

    /// `gate`, where there is one, windowed to `region` as the input is.
    pub fn windowed(gate: &ComptimeOption<Tile>, region: &Region) -> ComptimeOption<Tile> {
        gate.clone().map(|gate| gate.at(region))
    }

    /// Elements of a line.
    pub fn vector_size(&self) -> comptime_type!(usize) {
        self.input.vector_size()
    }

    /// Line `line` of the row the norm reads, held as `H`: the input is of element `E`.
    pub fn line<E: Float, H: Float, N: Size>(&self, line: usize) -> Vector<H, N> {
        let row = self.input.flat::<E, N>().read(line);
        match comptime!(self.order) {
            Some(GateOrder::GateThenNorm) => {
                let gate = self.gate_line::<E, N>(line);
                let gated = activation::gated::<N>(Vector::<f32, N>::cast_from(row), gate);
                Vector::<H, N>::cast_from(gated)
            }
            Some(GateOrder::NormThenGate) | None => Vector::<H, N>::cast_from(row),
        }
    }

    /// Line `line` of the norm, `scaled` in `f32`, as it is written: multiplied by the silu of the
    /// gate's where the gate comes last.
    pub fn written<E: Float, N: Size>(
        &self,
        line: usize,
        scaled: Vector<f32, N>,
    ) -> Vector<f32, N> {
        match comptime!(self.order) {
            Some(GateOrder::NormThenGate) => {
                activation::gated::<N>(scaled, self.gate_line::<E, N>(line))
            }
            Some(GateOrder::GateThenNorm) | None => scaled,
        }
    }

    fn gate_line<E: Float, N: Size>(&self, line: usize) -> Vector<f32, N> {
        #[comptime]
        match &self.gate {
            ComptimeOption::Some(gate) => {
                Vector::<f32, N>::cast_from(gate.flat::<E, N>().read(line))
            }
            ComptimeOption::None => panic!("a gated norm binds its gate"),
        }
    }
}
