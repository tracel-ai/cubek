use cubecl::{
    cmma::MmaDefinition,
    define_size,
    ir::{DeviceProperties, MatrixIdent, StorageType},
    prelude::*,
};

use crate::{
    MatrixLayout, StageIdent, TileSize,
    tile::{
        SharedTile, Tile, TileKind, TileKindExpand, TileScope,
        variants::instruction::mma::{MmaStageWriter, mma_fill_fragment, mma_load_strided},
    },
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
/// metadata the tile body uses. The matmul-level configuration that produced
/// these values lives in cubek-matmul as `MmaMatmul`.
#[derive(CubeType)]
pub struct MmaTile<N: Numeric> {
    pub fragment: MmaFragment<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub tile_size: TileSize,
    #[cube(comptime)]
    pub mma_io_config: MmaIOConfig,
}

#[derive(CubeType)]
pub enum MmaFragment<N: Numeric> {
    Lhs(Array<Vector<N, NL>>),
    Rhs(Array<Vector<N, NR>>),
    Acc(Array<Vector<N, NA>>),
}

/// Hardware-capability-driven choice of load/store methods for the MMA tile.
/// Determined once per `(device, dtypes)` and carried by the tile because the
/// fragment readers/writers branch on it.
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
    #[comptime] tile_size: TileSize,
) -> MmaDefinition<L, R, A> {
    MmaDefinition::new(
        tile_size.m() as usize,
        tile_size.n() as usize,
        tile_size.k() as usize,
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
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) -> Tile<L, Sc> {
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::A);

    Tile::from_kind(TileKind::new_Mma(MmaTile::<L> {
        fragment: MmaFragment::new_Lhs(Array::new(vector_count)),
        matrix_layout: layout,
        tile_size,
        mma_io_config,
    }))
}

#[cube]
pub fn mma_allocate_rhs<R: Numeric, L: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) -> Tile<R, Sc> {
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::B);

    Tile::from_kind(TileKind::new_Mma(MmaTile::<R> {
        fragment: MmaFragment::new_Rhs(Array::new(vector_count)),
        matrix_layout: layout,
        tile_size,
        mma_io_config,
    }))
}

#[cube]
pub fn mma_allocate_acc<A: Numeric, L: Numeric, R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) -> Tile<A, Sc> {
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::Accumulator);

    Tile::from_kind(TileKind::new_Mma(MmaTile::<A> {
        fragment: MmaFragment::new_Acc(Array::new(vector_count)),
        matrix_layout: layout,
        tile_size,
        mma_io_config,
    }))
}

#[cube]
impl<A: Numeric> MmaTile<A> {
    /// Executes `lhs · rhs`, accumulating into `self`. Each operand must
    /// carry the role its position requires (`Lhs`, `Rhs`, `Acc`).
    pub fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &MmaTile<L>, rhs: &MmaTile<R>) {
        match &lhs.fragment {
            MmaFragment::Lhs(lf) => match &rhs.fragment {
                MmaFragment::Rhs(rf) => match &mut self.fragment {
                    MmaFragment::Acc(af) => {
                        mma_execute(lf, rf, af, self.matrix_layout, self.tile_size);
                    }
                    MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                        panic!("Mma: expected Acc role for accumulator")
                    }
                },
                MmaFragment::Lhs(_) | MmaFragment::Acc(_) => {
                    panic!("Mma: expected Rhs role for rhs")
                }
            },
            MmaFragment::Rhs(_) | MmaFragment::Acc(_) => {
                panic!("Mma: expected Lhs role for lhs")
            }
        }
    }
}

#[cube]
impl<N: Numeric> MmaTile<N> {
    /// Copies into the mma fragment from `source`. Supported sources:
    /// `SharedMemory` (per-role load) and `None` (zero-init, Acc only).
    /// `L` / `R` / `A` are the matmul triple's role types — needed by the
    /// per-role load functions. When `self` is in role X, `N` substitutes
    /// for the X type and the other two are taken from the caller's
    /// generics.
    pub fn copy_from<SE: Numeric, SS: Size, L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
        &mut self,
        source: &Tile<SE, Sc>,
        #[comptime] _ident: StageIdent,
    ) {
        match &source.kind {
            TileKind::SharedTile(shared) => match &mut self.fragment {
                MmaFragment::Lhs(f) => mma_load_lhs_from_shared::<SE, SS, N, R, A>(
                    shared,
                    f,
                    self.matrix_layout,
                    self.tile_size,
                    self.mma_io_config,
                ),
                MmaFragment::Rhs(f) => mma_load_rhs_from_shared::<SE, SS, N, L, A>(
                    shared,
                    f,
                    self.matrix_layout,
                    self.tile_size,
                    self.mma_io_config,
                ),
                MmaFragment::Acc(f) => mma_load_acc_from_shared::<SE, SS, N, L, R>(
                    shared,
                    f,
                    self.matrix_layout,
                    self.tile_size,
                    self.mma_io_config,
                ),
            },
            TileKind::None => match &mut self.fragment {
                MmaFragment::Acc(f) => {
                    mma_load_acc_zeros::<N, L, R>(
                        f,
                        self.matrix_layout,
                        self.tile_size,
                        self.mma_io_config,
                    );
                }
                MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                    panic!("Mma zero-load only supported for Acc role")
                }
            },
            TileKind::Cmma(_)
            | TileKind::Mma(_)
            | TileKind::Register(_)
            | TileKind::PlaneVec(_)
            | TileKind::Interleaved(_)
            | TileKind::Unit(_)
            | TileKind::WhiteboxFragment(_)
            | TileKind::RowWise(_)
            | TileKind::Bounce(_)
            | TileKind::Stage(_)
            | TileKind::Partition(_)
            | TileKind::Pipelined(_) => panic!("MmaTile::copy_from: unsupported source variant"),
        }
    }

    /// Zero-init the mma fragment (Acc role only).
    pub fn init_zero<L: Numeric, R: Numeric>(&mut self) {
        match &mut self.fragment {
            MmaFragment::Acc(f) => {
                mma_load_acc_zeros::<N, L, R>(
                    f,
                    self.matrix_layout,
                    self.tile_size,
                    self.mma_io_config,
                );
            }
            MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                panic!("MmaTile::init_zero: only Acc role supported")
            }
        }
    }
}

// ===========================================================================
// Compute: matmul / load / write / zero-init
// ===========================================================================

#[cube]
pub fn mma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] _matrix_layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) {
    let def = MmaDefinition::<L, R, A>::new(
        tile_size.m() as usize,
        tile_size.n() as usize,
        tile_size.k() as usize,
    );
    let out_arr = def.execute(lhs, rhs, &*acc);
    let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num_vectors {
        acc[i] = out_arr[i];
    }
}

#[cube]
pub fn mma_load_lhs_from_shared<E: Numeric, ES: Size, L: Numeric, R: Numeric, A: Numeric>(
    shared: &SharedTile<E>,
    fragment: &mut Array<Vector<L, NL>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let shared = shared.view::<ES>();
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_load_strided(
        &shared,
        fragment,
        def,
        MatrixIdent::A,
        matrix_layout,
        tile_size,
        mma_io_config,
    );
}

#[cube]
pub fn mma_load_rhs_from_shared<E: Numeric, ES: Size, R: Numeric, L: Numeric, A: Numeric>(
    shared: &SharedTile<E>,
    fragment: &mut Array<Vector<R, NR>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let shared = shared.view::<ES>();
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_load_strided(
        &shared,
        fragment,
        def,
        MatrixIdent::B,
        matrix_layout,
        tile_size,
        mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_from_shared<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    shared: &SharedTile<E>,
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let shared = shared.view::<ES>();
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_load_strided(
        &shared,
        fragment,
        def,
        MatrixIdent::Accumulator,
        matrix_layout,
        tile_size,
        mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_zeros<A: Numeric, L: Numeric, R: Numeric>(
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let _ = (matrix_layout, mma_io_config);
    let def = make_mma_definition::<L, R, A>(tile_size);
    mma_fill_fragment::<A, NA, A, L, R, A>(
        &A::from_int(0),
        fragment,
        def,
        MatrixIdent::Accumulator,
    );
}

#[cube]
pub fn mma_write_to_shared<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    shared: &mut SharedTile<E>,
    fragment: &Array<Vector<A, NA>>,
    #[comptime] tile_size: TileSize,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let mut shared = shared.view::<ES>();
    let def = make_mma_definition::<L, R, A>(tile_size);
    let out_layout = comptime!(shared.layout);
    MmaStageWriter::store_fragment(
        &mut shared,
        fragment,
        &def,
        MatrixIdent::Accumulator,
        out_layout,
        tile_size.m(),
        mma_io_config,
    );
}
