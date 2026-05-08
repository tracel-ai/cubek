use cubecl::{
    cmma::MmaDefinition,
    define_size,
    ir::{DeviceProperties, MatrixIdent, StorageType},
    prelude::*,
};

use crate::{
    MatrixLayout, StageIdent, SwizzleModes, TileSize,
    tile::{
        Tile, TileKind, TileKindExpand, TileScope,
        variants::{
            kind::{Filled, Strided},
            mma::{MmaFragmentReader as _, MmaStageReader, MmaStageWriter},
            strided::SharedTile,
        },
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

    Tile::from_kind(TileKind::new_Mma(MmaTile::<L> {
        fragment: MmaFragment::new_Lhs(Array::new(vector_count)),
        matrix_layout: layout,
        config,
    }))
}

#[cube]
pub fn mma_allocate_rhs<R: Numeric, L: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) -> Tile<R, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::B);

    Tile::from_kind(TileKind::new_Mma(MmaTile::<R> {
        fragment: MmaFragment::new_Rhs(Array::new(vector_count)),
        matrix_layout: layout,
        config,
    }))
}

#[cube]
pub fn mma_allocate_acc<A: Numeric, L: Numeric, R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) -> Tile<A, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::Accumulator);

    Tile::from_kind(TileKind::new_Mma(MmaTile::<A> {
        fragment: MmaFragment::new_Acc(Array::new(vector_count)),
        matrix_layout: layout,
        config,
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
                        mma_execute(lf, rf, af, self.matrix_layout, self.config);
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
    pub fn copy_from<
        SE: Numeric,
        SS: Size,
        L: Numeric,
        R: Numeric,
        A: Numeric,
        Sc: TileScope,
        SIO: SliceVisibility,
    >(
        &mut self,
        source: &Tile<SE, Sc, SIO>,
        #[comptime] _ident: StageIdent,
    ) {
        match &source.kind {
            TileKind::SharedMemory(shared) => match &mut self.fragment {
                MmaFragment::Lhs(f) => mma_load_lhs_from_shared::<SE, SS, N, R, A, SIO>(
                    shared,
                    f,
                    self.matrix_layout,
                    self.config,
                ),
                MmaFragment::Rhs(f) => mma_load_rhs_from_shared::<SE, SS, N, L, A, SIO>(
                    shared,
                    f,
                    self.matrix_layout,
                    self.config,
                ),
                MmaFragment::Acc(f) => mma_load_acc_from_shared::<SE, SS, N, L, R, SIO>(
                    shared,
                    f,
                    self.matrix_layout,
                    self.config,
                ),
            },
            TileKind::None => match &mut self.fragment {
                MmaFragment::Acc(f) => {
                    mma_load_acc_zeros::<N, L, R>(f, self.matrix_layout, self.config);
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
            | TileKind::Bounce(_) => panic!("MmaTile::copy_from: unsupported source variant"),
        }
    }

    /// Zero-init the mma fragment (Acc role only).
    pub fn init_zero<L: Numeric, R: Numeric>(&mut self) {
        match &mut self.fragment {
            MmaFragment::Acc(f) => {
                mma_load_acc_zeros::<N, L, R>(f, self.matrix_layout, self.config);
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
    #[comptime] config: MmaMatmul,
) {
    let def = MmaDefinition::<L, R, A>::new(
        config.tile_size.m() as usize,
        config.tile_size.n() as usize,
        config.tile_size.k() as usize,
    );
    let out_arr = def.execute(lhs, rhs, acc);
    let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num_vectors {
        acc[i] = out_arr[i];
    }
}

#[cube]
pub fn mma_load_lhs_from_shared<
    E: Numeric,
    ES: Size,
    L: Numeric,
    R: Numeric,
    A: Numeric,
    IO: SliceVisibility,
>(
    shared: &SharedTile<E, IO>,
    fragment: &mut Array<Vector<L, NL>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let shared = shared.view::<ES>();
    let shared = shared.to_read_only();
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Strided>::load_fragment(
        &shared,
        fragment,
        def,
        MatrixIdent::A,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_load_rhs_from_shared<
    E: Numeric,
    ES: Size,
    R: Numeric,
    L: Numeric,
    A: Numeric,
    IO: SliceVisibility,
>(
    shared: &SharedTile<E, IO>,
    fragment: &mut Array<Vector<R, NR>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let shared = shared.view::<ES>();
    let shared = shared.to_read_only();
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Strided>::load_fragment(
        &shared,
        fragment,
        def,
        MatrixIdent::B,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_from_shared<
    E: Numeric,
    ES: Size,
    A: Numeric,
    L: Numeric,
    R: Numeric,
    IO: SliceVisibility,
>(
    shared: &SharedTile<E, IO>,
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let shared = shared.view::<ES>();
    let shared = shared.to_read_only();
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Strided>::load_fragment(
        &shared,
        fragment,
        def,
        MatrixIdent::Accumulator,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_zeros<A: Numeric, L: Numeric, R: Numeric>(
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Filled>::load_fragment::<A, NA, A, NA, L, R, A>(
        &A::from_int(0),
        fragment,
        def,
        MatrixIdent::Accumulator,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_write_to_shared<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    shared: &mut SharedTile<E, ReadWrite>,
    fragment: &Array<Vector<A, NA>>,
    #[comptime] config: MmaMatmul,
) {
    let mut shared = shared.view::<ES>();
    let def = make_mma_definition::<L, R, A>(config);
    let out_layout = comptime!(shared.layout);
    MmaStageWriter::store_fragment(
        &mut shared,
        fragment,
        def,
        MatrixIdent::Accumulator,
        out_layout,
        config.tile_size.m(),
        config.mma_io_config,
    );
}
