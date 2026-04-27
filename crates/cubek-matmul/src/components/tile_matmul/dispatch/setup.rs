use crate::components::resource::CubeDimResource;
use crate::components::tile_matmul::dispatch::DispatchTileMatmul;
use crate::components::tile_matmul::dispatch::config::DispatchConfig;
use crate::components::tile_matmul::{
    InterleavedMatmulConfig, MmaMatmulConfig, Plane, PlaneVecMatInnerProductConfig,
    RegisterMatmulConfig, SharedTileConfig, TileMatmulFamily,
};
use crate::definition::TilingBlueprint;
use crate::definition::{
    MatmulAvailabilityError, MatmulElems, MatmulSetupError, MatmulVectorSizes,
};
use cubecl::{
    features::{MmaConfig, Plane as PlaneFeature, TypeUsage},
    ir::{DeviceProperties, ElemType, FloatKind, StorageType},
    prelude::*,
};
use cubek_std::tile::mma::MmaIOConfig;
use cubek_std::{InvalidConfigError, MatrixLayout, TileSize};

impl TileMatmulFamily for DispatchTileMatmul {
    type Config = DispatchConfig;
    type Scope = Plane;
    type Matmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> =
        DispatchTileMatmul;

    fn requires_accelerator(&self) -> bool {
        match self {
            DispatchTileMatmul::Cmma | DispatchTileMatmul::Mma => true,
            DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => false,
        }
    }

    fn can_cast_stage_element(&self) -> bool {
        match self {
            DispatchTileMatmul::Cmma => false,
            DispatchTileMatmul::Mma
            | DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => true,
        }
    }

    fn should_swizzle<R: Runtime>(&self, client: &ComputeClient<R>) -> bool {
        match self {
            DispatchTileMatmul::Cmma => {
                // Unsupported
                false
            }
            DispatchTileMatmul::Mma => {
                // No alignment means swizzling can't be properly used, since it needs to be applied to
                // the address, and alignment guarantees the offset is aligned to the pattern repeat.
                client.properties().features.alignment
            }
            DispatchTileMatmul::Register => {
                // Selection isn't getting rid of all conflicts with the current load strategy, but does
                // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
                // optimal settings in the future.
                client.properties().features.alignment
            }
            DispatchTileMatmul::PlaneVec => {
                // Supported but need to find good settings for this tiling. Currently tuned for `ldmatrix`.
                // Need to profile at some point
                false
            }
            DispatchTileMatmul::Interleaved => {
                // Selection isn't getting rid of all conflicts with the current load strategy, but does
                // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
                // optimal settings in the future.
                client.properties().features.alignment
            }
        }
    }

    fn cubedim_resource(&self) -> Result<CubeDimResource, InvalidConfigError> {
        match self {
            DispatchTileMatmul::Cmma
            | DispatchTileMatmul::Mma
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => Ok(CubeDimResource::Planes(1)),
            DispatchTileMatmul::Register => Ok(CubeDimResource::Units(1)),
        }
    }

    fn expand_config(
        &self,
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<DispatchConfig, MatmulSetupError> {
        Ok(match self {
            DispatchTileMatmul::Cmma => DispatchConfig::Cmma(SharedTileConfig::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            )),
            DispatchTileMatmul::Mma => DispatchConfig::Mma(MmaMatmulConfig {
                shared: SharedTileConfig {
                    tile_size: blueprint.tiling_scheme.tile_size,
                    plane_dim: blueprint.plane_dim,
                    swizzle_modes: blueprint.swizzle_modes,
                },
                mma_io_config: MmaIOConfig::new(
                    device_props,
                    dtypes.lhs_stage,
                    dtypes.rhs_stage,
                    dtypes.acc_stage,
                ),
            }),
            DispatchTileMatmul::Register => {
                DispatchConfig::Register(RegisterMatmulConfig::from_shared_tile_config(
                    blueprint.lhs_layout,
                    blueprint.rhs_layout,
                    SharedTileConfig::new(
                        blueprint.tiling_scheme.tile_size,
                        blueprint.plane_dim,
                        blueprint.swizzle_modes,
                    ),
                ))
            }
            DispatchTileMatmul::PlaneVec => {
                DispatchConfig::PlaneVec(PlaneVecMatInnerProductConfig::new(
                    SharedTileConfig::new(
                        blueprint.tiling_scheme.tile_size,
                        blueprint.plane_dim,
                        blueprint.swizzle_modes,
                    ),
                    vector_sizes.lhs as u32,
                ))
            }
            DispatchTileMatmul::Interleaved => DispatchConfig::Interleaved(
                InterleavedMatmulConfig::from_shared_tile_config(SharedTileConfig::new(
                    blueprint.tiling_scheme.tile_size,
                    blueprint.plane_dim,
                    blueprint.swizzle_modes,
                )),
            ),
        })
    }

    fn is_supported<R: Runtime>(&self, client: &ComputeClient<R>, config: MmaConfig) -> bool {
        match self {
            DispatchTileMatmul::Cmma => client.properties().features.matmul.cmma.contains(&config),
            DispatchTileMatmul::Mma => client.properties().features.matmul.mma.contains(&config),
            DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => true,
        }
    }

    fn supported_sizes<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        let iters = match self {
            DispatchTileMatmul::Cmma => &client.properties().features.matmul.cmma,
            DispatchTileMatmul::Mma => &client.properties().features.matmul.mma,
            DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => return Vec::new(),
        };

        iters
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }

    fn validate_blueprint<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        match self {
            DispatchTileMatmul::Cmma => validate_cmma(client, blueprint, dtypes),
            DispatchTileMatmul::Mma => validate_mma(client, blueprint, dtypes),
            DispatchTileMatmul::Register => {
                validate_register(client, blueprint, dtypes, vector_sizes)
            }
            DispatchTileMatmul::PlaneVec => {
                validate_plane_vec(client, blueprint, dtypes, vector_sizes)
            }
            DispatchTileMatmul::Interleaved => {
                validate_interleaved(client, blueprint, dtypes, vector_sizes)
            }
        }
    }
}

fn validate_cmma<R: Runtime>(
    client: &ComputeClient<R>,
    blueprint: &TilingBlueprint,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs = dtypes.lhs_register;
    let rhs = dtypes.rhs_register;
    let acc = dtypes.acc_register;

    let size = blueprint.tiling_scheme.tile_size;
    if !client
        .properties()
        .features
        .matmul
        .cmma
        .contains(&MmaConfig {
            a_type: lhs,
            b_type: rhs,
            cd_type: acc,
            m: size.m(),
            k: size.k(),
            n: size.n(),
        })
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                lhs,
                rhs,
                output: acc,
                size: Some(TileSize::new(size.m(), size.n(), size.k())),
            },
        ));
    }

    if blueprint.swizzle_modes.has_swizzle() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "This tile matmul doesn't support swizzling",
        )));
    }

    Ok(())
}

fn validate_mma<R: Runtime>(
    client: &ComputeClient<R>,
    blueprint: &TilingBlueprint,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs = dtypes.lhs_register;
    let rhs = dtypes.rhs_register;
    let acc = dtypes.acc_register;

    let size = blueprint.tiling_scheme.tile_size;
    if !client
        .properties()
        .features
        .matmul
        .mma
        .contains(&MmaConfig {
            a_type: lhs,
            b_type: rhs,
            cd_type: acc,
            m: size.m(),
            k: size.k(),
            n: size.n(),
        })
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                lhs,
                rhs,
                output: acc,
                size: Some(TileSize::new(size.m(), size.n(), size.k())),
            },
        ));
    }

    Ok(())
}

fn validate_register<R: Runtime>(
    client: &ComputeClient<R>,
    blueprint: &TilingBlueprint,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(), MatmulSetupError> {
    check_types_available(client, dtypes, false)?;

    let m = blueprint.tiling_scheme.tile_size.m();
    let n = blueprint.tiling_scheme.tile_size.n();
    let k = blueprint.tiling_scheme.tile_size.k();

    let lhs = vector_sizes.lhs as u32;
    let rhs = vector_sizes.rhs as u32;
    let out = vector_sizes.out as u32;

    match blueprint.lhs_layout {
        MatrixLayout::RowMajor => {
            if !k.is_multiple_of(lhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Tile shape in vectorized axis k({k:?}) should be divisible by vector size lhs({lhs:?})"
                ))));
            }
        }
        MatrixLayout::ColMajor => {
            if !m.is_multiple_of(lhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Tile shape in vectorized axis m({m:?}) should be divisible by vector size lhs({lhs:?})"
                ))));
            }
        }
    }
    match blueprint.rhs_layout {
        MatrixLayout::RowMajor => {
            if !n.is_multiple_of(rhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Tile shape in vectorized axis n({n:?}) should be divisible by vector size rhs({rhs:?})"
                ))));
            }
        }
        MatrixLayout::ColMajor => {
            if !k.is_multiple_of(rhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Tile shape in vectorized axis k({k:?}) should be divisible by vector size rhs({rhs:?})"
                ))));
            }
        }
    }

    if !n.is_multiple_of(out) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Tile shape in vectorized axis n({n:?}) should be divisible by vector size out({out:?})"
        ))));
    }

    Ok(())
}

fn validate_plane_vec<R: Runtime>(
    client: &ComputeClient<R>,
    blueprint: &TilingBlueprint,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(), MatmulSetupError> {
    check_types_available(client, dtypes, true)?;

    if blueprint.lhs_layout != MatrixLayout::RowMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Only Row Major layout is supported for Lhs",
        )));
    }

    if blueprint.rhs_layout != MatrixLayout::ColMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Only Col Major layout is supported for Rhs",
        )));
    }

    let m = blueprint.tiling_scheme.tile_size.m();
    let n = blueprint.tiling_scheme.tile_size.n();
    let k = blueprint.tiling_scheme.tile_size.k();

    let lhs_vector = vector_sizes.lhs as u32;
    let rhs_vector = vector_sizes.rhs as u32;
    let out_vector = vector_sizes.out as u32;

    if m != 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Only m=1 is supported, got m={m:?}",
        ))));
    }

    if lhs_vector != rhs_vector {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Lhs and Rhs must have same vector size, got lhs={lhs_vector:?} and rhs={rhs_vector:?}",
        ))));
    }

    if k != blueprint.plane_dim * lhs_vector {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "k must be equal to plane_dim times vector size (of both lhs and rhs), got k={:?}, plane_dim={:?} vector_size={:?}",
            k, blueprint.plane_dim, lhs_vector
        ))));
    }

    if !n.is_multiple_of(out_vector) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "n must be divisible by out vector size, got n={n:?}, out_vector_size={out_vector:?}",
        ))));
    }

    Ok(())
}

fn validate_interleaved<R: Runtime>(
    client: &ComputeClient<R>,
    blueprint: &TilingBlueprint,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(), MatmulSetupError> {
    check_types_available(client, dtypes, false)?;

    let m = blueprint.tiling_scheme.tile_size.m();
    let n = blueprint.tiling_scheme.tile_size.n();
    let k = blueprint.tiling_scheme.tile_size.k();

    let plane_dim = blueprint.plane_dim;
    if !k.is_multiple_of(plane_dim) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "k must be divisible by plane_dim. Got k={:?}, plane_dim={:?}",
            k, plane_dim,
        ))));
    }

    let k_local = k / plane_dim;

    let lhs = vector_sizes.lhs as u32;
    let rhs = vector_sizes.rhs as u32;
    let out = vector_sizes.out as u32;

    match blueprint.lhs_layout {
        MatrixLayout::RowMajor => {
            if !k_local.is_multiple_of(lhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Local shape in vectorized axis k ({k_local:?}) should be divisible by vector size lhs ({lhs:?})"
                ))));
            }
        }
        MatrixLayout::ColMajor => {
            if !m.is_multiple_of(lhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Tile shape in vectorized axis m ({m:?}) should be divisible by vector size lhs ({lhs:?})"
                ))));
            }
        }
    }
    match blueprint.rhs_layout {
        MatrixLayout::RowMajor => {
            if !n.is_multiple_of(rhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Tile shape in vectorized axis n ({n:?}) should be divisible by vector size rhs ({rhs:?})"
                ))));
            }
        }
        MatrixLayout::ColMajor => {
            if !k_local.is_multiple_of(rhs) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Local shape in vectorized axis k ({k_local:?}) should be divisible by vector size rhs ({rhs:?})"
                ))));
            }
        }
    }

    if !n.is_multiple_of(out) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Tile shape in vectorized axis n ({n:?}) should be divisible by vector size out ({out:?})"
        ))));
    }

    Ok(())
}

fn check_types_available<R: Runtime>(
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
    require_plane_ops: bool,
) -> Result<(), MatmulSetupError> {
    if require_plane_ops
        && !client
            .properties()
            .features
            .plane
            .contains(PlaneFeature::Ops)
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneOpsUnavailable,
        ));
    }

    let lhs = normalize_flex32(dtypes.lhs_register);
    let rhs = normalize_flex32(dtypes.rhs_register);
    let output = normalize_flex32(dtypes.acc_register);

    if !(client
        .properties()
        .features
        .type_usage(lhs)
        .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(rhs)
            .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(output)
            .contains(TypeUsage::Arithmetic))
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output },
        ));
    }

    Ok(())
}

fn normalize_flex32(ty: StorageType) -> StorageType {
    match ty {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => ty,
    }
}
