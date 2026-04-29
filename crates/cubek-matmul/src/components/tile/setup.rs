use crate::components::tile::matmul::TileMatmul;
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
use cubek_std::tile::{
    Plane, Scope, Unit, cmma::CmmaMatmul, interleaved::InterleavedMatmul, mma::MmaMatmul,
    plane_vec_mat_inner_product::PlaneVecMatInnerProduct, register::RegisterMatmul,
};
use cubek_std::{CubeDimResource, InvalidConfigError, MatrixLayout, TileSize};

/// Selector for the tile-level matmul kind, used before per-kind config exists.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileMatmulKind {
    Cmma,
    Mma,
    Register,
    PlaneVec,
    Interleaved,
}

impl TileMatmulKind {
    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    pub fn requires_accelerator(&self) -> bool {
        match self {
            TileMatmulKind::Cmma | TileMatmulKind::Mma => true,
            TileMatmulKind::Register | TileMatmulKind::PlaneVec | TileMatmulKind::Interleaved => {
                false
            }
        }
    }

    /// Whether this matmul kind is able to cast on load/store from the stage.
    pub fn can_cast_stage_element(&self) -> bool {
        match self {
            TileMatmulKind::Cmma => false,
            TileMatmulKind::Mma
            | TileMatmulKind::Register
            | TileMatmulKind::PlaneVec
            | TileMatmulKind::Interleaved => true,
        }
    }

    /// Returns whether this tile matmul may benefit from swizzling.
    /// Used to determine the selection, since swizzling may require different stage sizes.
    pub fn should_swizzle<R: Runtime>(&self, client: &ComputeClient<R>) -> bool {
        match self {
            TileMatmulKind::Cmma => {
                // Unsupported
                false
            }
            TileMatmulKind::Mma => {
                // No alignment means swizzling can't be properly used, since it needs to be applied to
                // the address, and alignment guarantees the offset is aligned to the pattern repeat.
                client.properties().features.alignment
            }
            TileMatmulKind::Register => {
                // Selection isn't getting rid of all conflicts with the current load strategy, but does
                // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
                // optimal settings in the future.
                client.properties().features.alignment
            }
            TileMatmulKind::PlaneVec => {
                // Supported but need to find good settings for this tiling. Currently tuned for `ldmatrix`.
                // Need to profile at some point
                false
            }
            TileMatmulKind::Interleaved => {
                // Selection isn't getting rid of all conflicts with the current load strategy, but does
                // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
                // optimal settings in the future.
                client.properties().features.alignment
            }
        }
    }

    /// Returns the compute resources required to run this matmul.
    pub fn cubedim_resource(&self) -> Result<CubeDimResource, InvalidConfigError> {
        Ok(match self {
            TileMatmulKind::Cmma
            | TileMatmulKind::Mma
            | TileMatmulKind::PlaneVec
            | TileMatmulKind::Interleaved => Plane::default_resource(),
            TileMatmulKind::Register => Unit::default_resource(),
        })
    }

    /// Constructs the configuration based on the matmul problem, selection, and vector sizes.
    pub fn expand_config(
        &self,
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<TileMatmul, MatmulSetupError> {
        Ok(match self {
            TileMatmulKind::Cmma => TileMatmul::Cmma(CmmaMatmul::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            )),
            TileMatmulKind::Mma => TileMatmul::Mma(MmaMatmul {
                tile_size: blueprint.tiling_scheme.tile_size,
                plane_dim: blueprint.plane_dim,
                swizzle_modes: blueprint.swizzle_modes,
                mma_io_config: MmaIOConfig::new(
                    device_props,
                    dtypes.lhs_stage,
                    dtypes.rhs_stage,
                    dtypes.acc_stage,
                ),
            }),
            TileMatmulKind::Register => TileMatmul::Register(RegisterMatmul::new(
                blueprint.lhs_layout,
                blueprint.rhs_layout,
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            )),
            TileMatmulKind::PlaneVec => TileMatmul::PlaneVec(PlaneVecMatInnerProduct::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
                vector_sizes.lhs as u32,
            )),
            TileMatmulKind::Interleaved => TileMatmul::Interleaved(InterleavedMatmul::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            )),
        })
    }

    /// Returns whether a tile configuration is supported.
    pub fn is_supported<R: Runtime>(&self, client: &ComputeClient<R>, config: MmaConfig) -> bool {
        match self {
            TileMatmulKind::Cmma => client.properties().features.matmul.cmma.contains(&config),
            TileMatmulKind::Mma => client.properties().features.matmul.mma.contains(&config),
            TileMatmulKind::Register | TileMatmulKind::PlaneVec | TileMatmulKind::Interleaved => {
                true
            }
        }
    }

    /// Returns all sizes supported for these types, if any.
    pub fn supported_sizes<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        let iters = match self {
            TileMatmulKind::Cmma => &client.properties().features.matmul.cmma,
            TileMatmulKind::Mma => &client.properties().features.matmul.mma,
            TileMatmulKind::Register | TileMatmulKind::PlaneVec | TileMatmulKind::Interleaved => {
                return Vec::new();
            }
        };

        iters
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }

    pub fn validate_blueprint<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        match self {
            TileMatmulKind::Cmma => validate_cmma(client, blueprint, dtypes),
            TileMatmulKind::Mma => validate_mma(client, blueprint, dtypes),
            TileMatmulKind::Register => validate_register(client, blueprint, dtypes, vector_sizes),
            TileMatmulKind::PlaneVec => validate_plane_vec(client, blueprint, dtypes, vector_sizes),
            TileMatmulKind::Interleaved => {
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
