use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::StridedTile;

use cubecl::std::tensor::layout::Coordinates;

use crate::components::tile::softmax::{
    FragmentMask, FragmentMaskExpand, Softmax, SoftmaxConfig as _, SoftmaxLayout,
    SoftmaxLayoutExpand,
};

#[derive(CubeType)]
/// Mask tile for Tile Attention
/// It is an additive mask, which means the result of apply should be added, not multiplied
pub enum MaskTile<M: FragmentMask> {
    /// When a mask tensor is supplied. Also contains a logical part
    Materialized(MaterializedTileMask<M>),
    /// When no mask tensor is supplied. Used for out of bounds and causal mask
    Logical(LogicalTileMask<M::Layout>),
}

#[cube]
impl<M: FragmentMask> MaskTile<M> {
    pub fn new<F: Float, SMX: Softmax<F, Mask = M, ScoreLayout = M::Layout>>(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] config: SMX::Config,
    ) -> MaskTile<M> {
        let logical_mask = LogicalTileMask::<SMX::ScoreLayout> {
            logical_iter_origin: LogicalIterOrigin::init(),
            causal: config.causal_mask(),
            out_of_bounds,
            fragment_layout: SMX::layout(config),
        };

        if config.materialized_mask() {
            MaskTile::new_Materialized(MaterializedTileMask::<M> {
                fragment: SMX::allocate_mask(config),
                logical_mask,
                config,
            })
        } else {
            MaskTile::new_Logical(logical_mask)
        }
    }

    /// Loads the mask data into the fragment, if a tile is given, otherwise only
    /// updates the logical mask
    pub fn update<E: Numeric>(
        &mut self,
        new_origin: Coords2d,
        tile: ComptimeOption<StridedTile<E>>,
    ) {
        match self {
            MaskTile::Materialized(materialized_tile_mask) => {
                materialized_tile_mask
                    .logical_mask
                    .update_origin(new_origin);

                materialized_tile_mask.update_tile(tile.unwrap())
            }
            MaskTile::Logical(logical_tile_mask) => logical_tile_mask.update_origin(new_origin),
        }
    }
}

#[derive(CubeType)]
/// Gives the origin of the logical mask, which is updated when changing partition or tile within partition
pub struct LogicalIterOrigin {
    row: RuntimeCell<u32>,
    col: RuntimeCell<u32>,
}

#[cube]
impl LogicalIterOrigin {
    fn init() -> LogicalIterOrigin {
        LogicalIterOrigin {
            row: RuntimeCell::new(0),
            col: RuntimeCell::new(0),
        }
    }

    fn read(&self) -> Coords2d {
        (self.row.read(), self.col.read())
    }

    fn update(&mut self, new: Coords2d) {
        self.row.store(new.0);
        self.col.store(new.1);
    }
}

#[derive(CubeType)]
pub struct LogicalTileMask<F: SoftmaxLayout> {
    // Indicates where the logical mask currently starts
    logical_iter_origin: LogicalIterOrigin,
    #[cube(comptime)]
    // Whether to apply causal mask
    causal: bool,
    // Coordinates over which softmax is out of bounds, corresponds to seq_q, seq_kv of the problem
    out_of_bounds: ComptimeOption<Coords2d>,
    // Allows mapping local position of a unit to its absolute position
    fragment_layout: F,
}

#[cube]
impl<F: SoftmaxLayout> LogicalTileMask<F> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let pos_in_tile = self.fragment_layout.absolute_pos(local_pos);

        let pos = Coords2d::add(self.logical_iter_origin.read(), pos_in_tile);

        let causal_masked = self.causal && pos.0 < pos.1;

        #[comptime]
        let oob_masked = match self.out_of_bounds {
            ComptimeOption::Some(bounds) => !Coords2d::is_in_bounds(&pos, &bounds),
            ComptimeOption::None => false,
        };

        causal_masked || oob_masked
    }

    pub fn update_origin(&mut self, new_origin: Coords2d) {
        self.logical_iter_origin.update(new_origin);
    }
}

#[derive(CubeType)]
pub struct MaterializedTileMask<M: FragmentMask> {
    fragment: M,
    logical_mask: LogicalTileMask<M::Layout>,
    #[cube(comptime)]
    config: SMX::Config,
}

#[cube]
impl<M: FragmentMask> MaterializedTileMask<M> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let logical_masked = self.logical_mask.should_mask(local_pos);
        let materialized_masked = self.fragment.should_mask(local_pos);

        logical_masked || materialized_masked
    }

    pub fn update_tile<MSK: Numeric>(&mut self, tile: StridedTile<MSK>) {
        SMX::load_mask(&tile, &mut self.fragment, self.config);
    }
}

#[cube]
impl<M: FragmentMask> FragmentMask for MaskTile<M> {
    type Layout = M::Layout;

    fn should_mask(&self, local_pos: (u32, u32)) -> bool {
        match self {
            MaskTile::Materialized(materialized_tile_mask) => {
                materialized_tile_mask.should_mask(local_pos)
            }
            MaskTile::Logical(logical_tile_mask) => logical_tile_mask.should_mask(local_pos),
        }
    }
}
