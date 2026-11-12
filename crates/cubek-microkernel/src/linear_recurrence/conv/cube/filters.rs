use super::past_taps::PastTaps;
use crate::linear_recurrence::CHANNEL;
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};
use cubek_std::microkernel::{ArithmeticInstruction, UnitWork};

/// The logistic function.
#[cube]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// The silu of `x`.
#[cube]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// The depthwise filters of every channel and, where there is one, each channel's bias.
#[derive(CubeType)]
pub struct CausalFilters {
    filters: Tile,
    bias: ComptimeOption<Tile>,
    #[cube(comptime)]
    taps: usize,
}

impl CausalFilters {
    /// What a unit does per line of channels over `taps` taps: the taps and their products held in
    /// registers.
    pub fn unit_work(taps: usize) -> UnitWork {
        UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: (2 * taps + 8) as u32,
        }
    }
}

#[cube]
impl CausalFilters {
    pub fn new(filters: &Tile, bias: &ComptimeOption<Tile>, #[comptime] taps: usize) -> Self {
        CausalFilters {
            filters: filters.clone(),
            bias: bias.clone(),
            taps,
        }
    }

    /// The `line`-th line of the convolved channels: the newest value and the taps behind it times
    /// the filters, plus the bias, then the silu.
    pub fn convolved_line<E: Float, N: Size>(
        &self,
        past: &PastTaps,
        newest: Vector<E, N>,
        line: usize,
    ) -> Vector<f32, N> {
        let read_columns = comptime!(self.taps - 1);
        let first = line * N::value();
        let mut sums =
            Vector::<f32, N>::cast_from(newest) * self.weights::<E, N>(first, read_columns);
        #[unroll]
        for column in 0..read_columns {
            let behind = Vector::<f32, N>::cast_from(past.behind::<E, N>(column, line));
            sums += behind * self.weights::<E, N>(first, column);
        }
        #[comptime]
        match &self.bias {
            ComptimeOption::Some(bias) => {
                sums += Vector::<f32, N>::cast_from(bias.flat::<E, N>().read(line))
            }
            ComptimeOption::None => {}
        }
        Self::activated::<E, N>(sums)
    }

    fn weights<E: Float, N: Size>(&self, first: usize, tap: usize) -> Vector<f32, N> {
        let width = N::value();
        let mut weights = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for lane in 0..width {
            let row = self.filters.within(CHANNEL, first + lane, first + lane + 1);
            let weight = row.flat::<E, Const<1>>().read(tap).extract(0usize);
            weights.insert(lane, f32::cast_from(weight));
        }
        weights
    }

    /// The silu of `sums`, rounded to `E` before and after as composed ops hold them.
    fn activated<E: Float, N: Size>(sums: Vector<f32, N>) -> Vector<f32, N> {
        let width = N::value();
        let mixed = Vector::<f32, N>::cast_from(Vector::<E, N>::cast_from(sums));
        let mut convolved = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for lane in 0..width {
            convolved.insert(lane, silu(mixed.extract(lane)));
        }
        Vector::<f32, N>::cast_from(Vector::<E, N>::cast_from(convolved))
    }
}
