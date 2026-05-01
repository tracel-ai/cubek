use cubecl;
use cubecl::std::tensor::layout::Coordinates;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::{
    Mask, MaskExpand, MaskLayout, Plane, StridedTile, Tile, allocate_local_tile,
    allocate_unit_tile, mask_layout_absolute_pos,
};

/// Comptime configuration that drives [`MaskTile::new`] and
/// `MaskTile::should_mask`. Built attention-side from a `TileSoftmax`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MaskConfig {
    pub layout: MaskLayout,
    pub causal: bool,
    pub materialized: bool,
}

#[derive(CubeType)]
/// Mask tile for Tile Attention. It is an additive mask: the result of `apply`
/// should be added, not multiplied.
pub enum MaskTile<F: Float> {
    /// When a mask tensor is supplied. Also contains a logical part.
    Materialized(MaterializedTileMask<F>),
    /// When no mask tensor is supplied. Used for out-of-bounds and causal mask.
    Logical(LogicalTileMask),
}

#[cube]
impl<F: Float> MaskTile<F> {
    pub fn new(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] config: MaskConfig,
    ) -> MaskTile<F> {
        let logical_mask = LogicalTileMask::new(config, out_of_bounds);

        if comptime!(config.materialized) {
            let fragment: Tile<F, Plane, ReadWrite> = match comptime!(config.layout) {
                MaskLayout::Unit(l) => allocate_unit_tile::<F, Plane>(comptime!(l)),
                MaskLayout::Local(l) => allocate_local_tile::<F, Plane>(comptime!(l)),
            };
            MaskTile::new_Materialized(MaterializedTileMask::<F> {
                fragment,
                logical_mask,
            })
        } else {
            MaskTile::new_Logical(logical_mask)
        }
    }

    /// Loads the mask data into the fragment if a tile is given; otherwise
    /// only updates the logical mask's origin.
    pub fn update<E: Numeric, ES: Size>(
        &mut self,
        new_origin: Coords2d,
        tile: ComptimeOption<StridedTile<E, ES>>,
    ) {
        match self {
            MaskTile::Materialized(m) => {
                m.logical_mask.update_origin(new_origin);
                m.update_tile(tile.unwrap());
            }
            MaskTile::Logical(l) => l.update_origin(new_origin),
        }
    }
}

#[cube]
impl<F: Float> Mask for MaskTile<F> {
    fn should_mask(&self, local_pos: Coords2d) -> bool {
        match self {
            MaskTile::Materialized(m) => m.should_mask(local_pos),
            MaskTile::Logical(l) => l.should_mask(local_pos),
        }
    }
}

#[derive(CubeType)]
/// Origin of the logical mask, updated when changing partition or tile within
/// partition.
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
pub struct LogicalTileMask {
    logical_iter_origin: LogicalIterOrigin,
    #[cube(comptime)]
    causal: bool,
    out_of_bounds: ComptimeOption<Coords2d>,
    #[cube(comptime)]
    fragment_layout: MaskLayout,
}

#[cube]
impl LogicalTileMask {
    pub fn new(
        #[comptime] config: MaskConfig,
        out_of_bounds: ComptimeOption<Coords2d>,
    ) -> LogicalTileMask {
        LogicalTileMask {
            logical_iter_origin: LogicalIterOrigin::init(),
            causal: comptime!(config.causal),
            out_of_bounds,
            fragment_layout: comptime!(config.layout),
        }
    }

    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let pos_in_tile = mask_layout_absolute_pos(self.fragment_layout, local_pos);

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
pub struct MaterializedTileMask<F: Float> {
    fragment: Tile<F, Plane, ReadWrite>,
    logical_mask: LogicalTileMask,
}

#[cube]
impl<F: Float> MaterializedTileMask<F> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let logical_masked = self.logical_mask.should_mask(local_pos);
        let materialized_masked = self.fragment.should_mask(local_pos);

        logical_masked || materialized_masked
    }

    pub fn update_tile<MSK: Numeric, MSKS: Size>(&mut self, tile: StridedTile<MSK, MSKS>) {
        self.fragment
            .load_mask_from_strided_tile::<MSK, MSKS>(&tile);
    }
}
