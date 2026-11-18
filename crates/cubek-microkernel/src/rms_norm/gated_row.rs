use crate::activation;
use crate::rms_norm::GateOrder;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// A gate's tile and where it enters the norm, held together so a gated row cannot carry one
/// without the other.
#[derive(CubeType, Clone)]
struct Gate {
    tile: Tile,
    #[cube(comptime)]
    order: GateOrder,
}

impl core::fmt::Debug for Gate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gate").finish_non_exhaustive()
    }
}

#[cube]
impl Gate {
    /// The gate a launch binds: `tile` entering at `order`.
    fn bound(
        tile: ComptimeOption<Tile>,
        #[comptime] order: Option<GateOrder>,
    ) -> ComptimeOption<Gate> {
        #[comptime]
        match tile {
            ComptimeOption::Some(tile) => ComptimeOption::new_Some(Gate {
                tile,
                order: comptime!(order.expect("a bound gate tile has an order")),
            }),
            ComptimeOption::None => Gate::unbound(order),
        }
    }

    /// No gate, where a launch binds no tile and so states no order.
    fn unbound(#[comptime] order: Option<GateOrder>) -> ComptimeOption<Gate> {
        comptime!(assert!(order.is_none(), "a gate order has a bound tile"));
        ComptimeOption::new_None()
    }

    /// This gate windowed to `region`.
    fn at(&self, region: &Region) -> Gate {
        Gate {
            tile: self.tile.at(region),
            order: comptime!(self.order),
        }
    }

    /// `row`'s line `line`, of element `E`, held as `Held`: multiplied by the silu of the gate's
    /// where the gate comes first.
    fn before_norm<E: Float, Held: Float, N: Size>(
        &self,
        row: Vector<E, N>,
        line: usize,
    ) -> Vector<Held, N> {
        match comptime!(self.order) {
            GateOrder::GateThenNorm => Vector::<Held, N>::cast_from(activation::silu_gated::<N>(
                Vector::<f32, N>::cast_from(row),
                self.line::<E, N>(line),
            )),
            GateOrder::NormThenGate => Vector::<Held, N>::cast_from(row),
        }
    }

    /// `scaled`, the norm's line `line`, multiplied by the silu of the gate's where the gate comes
    /// last.
    fn after_norm<E: Float, N: Size>(&self, scaled: Vector<f32, N>, line: usize) -> Vector<f32, N> {
        match comptime!(self.order) {
            GateOrder::NormThenGate => activation::silu_gated::<N>(scaled, self.line::<E, N>(line)),
            GateOrder::GateThenNorm => scaled,
        }
    }

    /// Line `line` of the gate, its tile of element `E`, as `f32`.
    fn line<E: Float, N: Size>(&self, line: usize) -> Vector<f32, N> {
        Vector::<f32, N>::cast_from(self.tile.flat::<E, N>().read(line))
    }
}

/// A row of the norm's input and, where the norm is gated, of its [`Gate`]. It yields the lines
/// the norm sums and scales, gated where the gate comes first, and gates the lines the norm writes
/// where it comes last.
#[derive(CubeType)]
pub struct GatedRow {
    input: Tile,
    gate: ComptimeOption<Gate>,
}

impl core::fmt::Debug for GatedRow {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GatedRow").finish_non_exhaustive()
    }
}

#[cube]
impl GatedRow {
    /// The row of `input`, gated by `gate` entering at `order`. A launch binds a gate exactly
    /// where its blueprint states an order.
    pub fn new(
        input: &Tile,
        gate: ComptimeOption<Tile>,
        #[comptime] order: Option<GateOrder>,
    ) -> Self {
        GatedRow {
            input: input.clone(),
            gate: Gate::bound(gate, order),
        }
    }

    /// This row and its gate windowed to `region`.
    pub fn at(&self, region: &Region) -> GatedRow {
        let gate = {
            #[comptime]
            match &self.gate {
                ComptimeOption::Some(gate) => ComptimeOption::new_Some(gate.at(region)),
                ComptimeOption::None => ComptimeOption::new_None(),
            }
        };
        GatedRow {
            input: self.input.at(region),
            gate,
        }
    }

    /// Elements of a line.
    pub fn vector_size(&self) -> comptime_type!(usize) {
        self.input.vector_size()
    }

    /// Line `line` of the row the norm reads, held as `Held`: the input is of element `E`.
    pub fn line<E: Float, Held: Float, N: Size>(&self, line: usize) -> Vector<Held, N> {
        let row = self.input.flat::<E, N>().read(line);
        #[comptime]
        match &self.gate {
            ComptimeOption::Some(gate) => gate.before_norm::<E, Held, N>(row, line),
            ComptimeOption::None => Vector::<Held, N>::cast_from(row),
        }
    }

    /// Line `line` of the norm, `scaled` in `f32`, as it is written: multiplied by the silu of the
    /// gate's where the gate comes last.
    pub fn written<E: Float, N: Size>(
        &self,
        line: usize,
        scaled: Vector<f32, N>,
    ) -> Vector<f32, N> {
        #[comptime]
        match &self.gate {
            ComptimeOption::Some(gate) => gate.after_norm::<E, N>(scaled, line),
            ComptimeOption::None => scaled,
        }
    }
}
