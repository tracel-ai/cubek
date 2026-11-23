use super::CO;
use cubecl::tile::procedural::{MemoryValues, Procedural};
use cubecl::tile::{Partitioning, Region, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};

/// The values of a [`ConvBias::PerChannel`](super::ConvBias) read by output channel, where a
/// plane adds them to the cells it wrote.
#[derive(CubeType)]
pub struct BiasTile {
    values: Tile,
}

#[cube]
impl BiasTile {
    pub fn new<EO: Float>(bias: &TileTensor<'_>, partitioning: &Partitioning) -> Self {
        let values = bias.tile(partitioning).values();
        let space = comptime!(partitioning.space().subspace(&[CO]));
        BiasTile {
            values: Procedural::new::<EO, MemoryValues>(space, values).tile_in::<EO>(partitioning),
        }
    }

    /// Waits for the whole cube where there is a bias, before planes read back an output the
    /// cube wrote together.
    pub fn wait_for_cube(bias: &ComptimeOption<BiasTile>) {
        #[comptime]
        match bias {
            ComptimeOption::Some(_) => sync_cube(),
            ComptimeOption::None => {}
        }
    }

    /// Adds `bias`, where there is one, to the cells of `out` in `plane`'s window, once every one
    /// of them is written: the sum is finished, partials merged, before a channel's value joins it.
    pub fn add_to<EO: Float>(bias: &ComptimeOption<BiasTile>, out: &Tile, plane: &Region) {
        #[comptime]
        match bias {
            ComptimeOption::Some(bias) => bias.add::<EO>(out, plane),
            ComptimeOption::None => {}
        }
    }
}

#[cube]
impl BiasTile {
    fn add<EO: Float>(&self, out: &Tile, plane: &Region) {
        sync_plane();
        out.at(plane)
            .along(CO)
            .scale_add::<EO>(EO::from_int(1), &self.values.at(plane));
    }
}
