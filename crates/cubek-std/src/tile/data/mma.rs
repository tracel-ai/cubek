use cubecl::{
    cmma::MmaDefinition,
    define_size,
    ir::{DeviceProperties, MatrixIdent, StorageType},
    prelude::*,
};

use crate::{
    MatrixLayout, SwizzleModes, TileSize,
    tile::{Tile, TileScope},
};

// Fragment inner vector sizes for the three MMA roles. Bound at allocation time
// via `mma_register_vector_sizes` to match the hardware's `def.vector_size(...)`
// for each role — these are independent of the outer Tile enum's stage vector `V`.
define_size!(pub NL);
define_size!(pub NR);
define_size!(pub NA);

/// Single MMA tile carrier. The role (Lhs / Rhs / Acc) lives inside
/// [`MmaFragment`] because each role's fragment uses a different inner vector
/// size (`NL` / `NR` / `NA`); the outer carrier holds the shared comptime
/// metadata.
#[derive(CubeType)]
pub struct MmaTile<N: Numeric> {
    pub fragment: MmaFragment<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: MmaMatmul,
}

#[derive(CubeType)]
pub enum MmaFragment<N: Numeric> {
    Lhs(Array<Vector<N, NL>>),
    Rhs(Array<Vector<N, NR>>),
    Acc(Array<Vector<N, NA>>),
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub mma_io_config: MmaIOConfig,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaIOConfig {
    pub lhs_load_method: LoadMethod,
    pub rhs_load_method: LoadMethod,
    pub acc_load_method: LoadMethod,
    pub store_method: StoreMethod,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadMethod {
    Manual,
    LoadMatrix,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StoreMethod {
    Manual,
    StoreMatrix,
}

impl MmaIOConfig {
    pub fn new(
        device_props: &DeviceProperties,
        lhs_stage: StorageType,
        rhs_stage: StorageType,
        acc_stage: StorageType,
    ) -> Self {
        Self {
            lhs_load_method: load_method(device_props, lhs_stage),
            rhs_load_method: load_method(device_props, rhs_stage),
            acc_load_method: load_method(device_props, acc_stage),
            store_method: store_method(device_props, acc_stage),
        }
    }

    pub fn load_method(&self, ident: MatrixIdent) -> LoadMethod {
        match ident {
            MatrixIdent::A => self.lhs_load_method,
            MatrixIdent::B => self.rhs_load_method,
            MatrixIdent::Accumulator => self.acc_load_method,
        }
    }

    pub fn store_method(&self) -> StoreMethod {
        self.store_method
    }
}

fn load_method(device_props: &DeviceProperties, dtype: StorageType) -> LoadMethod {
    if !matches!(dtype, StorageType::Packed(_, _))
        && device_props.features.matmul.ldmatrix.contains(&dtype)
    {
        LoadMethod::LoadMatrix
    } else {
        LoadMethod::Manual
    }
}

fn store_method(device_props: &DeviceProperties, dtype: StorageType) -> StoreMethod {
    if !matches!(dtype, StorageType::Packed(_, _))
        && device_props.features.matmul.stmatrix.contains(&dtype)
    {
        StoreMethod::StoreMatrix
    } else {
        StoreMethod::Manual
    }
}

#[cube]
fn make_mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmul,
) -> MmaDefinition<L, R, A> {
    MmaDefinition::new(
        config.tile_size.m() as usize,
        config.tile_size.n() as usize,
        config.tile_size.k() as usize,
    )
}

#[cube]
#[allow(unused_variables)]
pub fn mma_register_vector_sizes<L: Numeric, R: Numeric, A: Numeric>(def: MmaDefinition<L, R, A>) {
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_size_acc = def.vector_size(MatrixIdent::Accumulator);
    intrinsic!(|scope| {
        scope.register_size::<NL>(vector_size_a);
        scope.register_size::<NR>(vector_size_b);
        scope.register_size::<NA>(vector_size_acc);
    });
}

#[cube]
pub fn mma_allocate_lhs<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) -> Tile<L, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::A);

    Tile::new_Mma(MmaTile::<L> {
        fragment: MmaFragment::new_Lhs(Array::new(vector_count)),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn mma_allocate_rhs<R: Numeric, L: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) -> Tile<R, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::B);

    Tile::new_Mma(MmaTile::<R> {
        fragment: MmaFragment::new_Rhs(Array::new(vector_count)),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn mma_allocate_acc<A: Numeric, L: Numeric, R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) -> Tile<A, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::Accumulator);

    Tile::new_Mma(MmaTile::<A> {
        fragment: MmaFragment::new_Acc(Array::new(vector_count)),
        matrix_layout: layout,
        config,
    })
}
