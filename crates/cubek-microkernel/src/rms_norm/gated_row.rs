use crate::activation;
use crate::rms_norm::GateOrder;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// A row of the norm's input and, where the norm is gated, of its gate. It yields the lines the
/// norm sums and scales, gated where the gate comes first, and gates the lines the norm writes
/// where it comes last.
#[derive(CubeType)]
pub struct GatedRow {
    input: Tile,
    gate: ComptimeOption<Tile>,
    #[cube(comptime)]
    order: Option<GateOrder>,
}

impl core::fmt::Debug for GatedRow {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GatedRow").finish_non_exhaustive()
    }
}

#[cube]
impl GatedRow {
    /// The row of `input` and, where the norm is gated, `gate`, both windowed to the owner's
    /// run.
    pub fn new(
        input: &Tile,
        gate: &ComptimeOption<Tile>,
        #[comptime] order: Option<GateOrder>,
    ) -> Self {
        GatedRow {
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
            Some(GateOrder::GateThenNorm) => self.gated_line::<E, H, N>(row, line),
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
            Some(GateOrder::NormThenGate) => self.gated_scaled::<E, N>(scaled, line),
            Some(GateOrder::GateThenNorm) | None => scaled,
        }
    }

    fn gated_line<E: Float, H: Float, N: Size>(
        &self,
        row: Vector<E, N>,
        line: usize,
    ) -> Vector<H, N> {
        let gate = self.gate_line::<E, N>(line);
        let gated = activation::gated::<N>(Vector::<f32, N>::cast_from(row), gate);
        Vector::<H, N>::cast_from(gated)
    }

    fn gated_scaled<E: Float, N: Size>(
        &self,
        scaled: Vector<f32, N>,
        line: usize,
    ) -> Vector<f32, N> {
        activation::gated::<N>(scaled, self.gate_line::<E, N>(line))
    }

    fn gate_line<E: Float, N: Size>(&self, line: usize) -> Vector<f32, N> {
        #[comptime]
        match &self.gate {
            ComptimeOption::Some(gate) => Self::line_as_f32::<E, N>(gate, line),
            ComptimeOption::None => panic!("a gated norm binds its gate"),
        }
    }

    fn line_as_f32<E: Float, N: Size>(tile: &Tile, line: usize) -> Vector<f32, N> {
        Vector::<f32, N>::cast_from(tile.flat::<E, N>().read(line))
    }
}
