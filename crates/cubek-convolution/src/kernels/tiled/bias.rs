use super::CO;
use cubecl::tile::procedural::{MemoryValues, Procedural};
use cubecl::tile::{Dimension, Partitioning, Region, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};

/// The values of a [`ConvBias::PerChannel`](super::ConvBias) read by output channel, where a
/// plane adds them to the cells it wrote.
#[derive(CubeType)]
pub struct BiasTile {
    values: Tile,
}

#[cube]
impl BiasTile {
    pub fn new<EO: Float>(
        bias: &TileTensor<'_>,
        partitioning: &Partitioning,
        #[comptime] dimensions: Vec<Dimension>,
    ) -> Self {
        let values = bias.tile(partitioning).values();
        let space = comptime!(partitioning.space().subspace(&dimensions));
        BiasTile {
            values: Procedural::new::<EO, MemoryValues>(space, values).tile_in::<EO>(partitioning),
        }
    }

    /// Waits for the whole cube, storage memory included, where there is a bias, before planes
    /// read back an output the cube wrote together.
    pub fn wait_for_cube(bias: &ComptimeOption<BiasTile>) {
        #[comptime]
        match bias {
            ComptimeOption::Some(_) => sync_storage(),
            ComptimeOption::None => {}
        }
    }

    /// Waits for the plane, storage memory included, where there is a bias, before it reads back
    /// the output cells its units wrote. Every plane reaches it, inside the output or not.
    pub fn wait_for_plane(bias: &ComptimeOption<BiasTile>) {
        #[comptime]
        match bias {
            ComptimeOption::Some(_) => sync_plane_storage(),
            ComptimeOption::None => {}
        }
    }

    /// [`add_to`](Self::add_to) for a plane whose box starts `inside` the output: one past its edge
    /// wrote nothing and owns no cell to add to.
    pub fn add_inside<EO: Float>(
        bias: &ComptimeOption<BiasTile>,
        out: &Tile,
        plane: &Region,
        inside: bool,
    ) {
        if inside {
            Self::add_to::<EO>(bias, out, plane);
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
        out.at(plane)
            .along(CO)
            .scale_add::<EO>(EO::from_int(1), &self.values.at(plane));
    }
}
