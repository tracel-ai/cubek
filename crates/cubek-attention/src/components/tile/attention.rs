use std::{fmt::Debug, hash::Hash};

use cubecl::features::MmaConfig;
use cubecl::{ir::DeviceProperties, ir::VectorSize};
use cubek_matmul::definition::MatmulAvailabilityError;
use cubek_std::tile::{BounceConfig, InnerLayout, MaskLayout, SoftmaxKind};
use cubek_std::{CubeDimResource, InvalidConfigError};

use crate::components::tile::matmul::AttentionTileMatmul;
use crate::definition::{
    AttentionAvailabilityError, AttentionBlueprint, AttentionElems, AttentionSetupError,
    AttentionTileSize, AttentionVectorSizes,
};

/// Comptime configuration for the entire tile-level attention. Bundles the
/// matmul choices for the score and value matmuls together with the
/// attention-domain knobs (causal, materialized mask, etc.) used by mask
/// construction and workspace allocation.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TileAttention {
    pub score_matmul: AttentionTileMatmul,
    pub value_matmul: AttentionTileMatmul,
    pub tile_size: AttentionTileSize,
    pub plane_dim: u32,
    pub num_planes: u32,
    pub inner_layout: InnerLayout,
    pub causal_mask: bool,
    pub materialized_mask: bool,
}

impl TileAttention {
    pub fn score_matmul(&self) -> AttentionTileMatmul {
        self.score_matmul
    }

    pub fn value_matmul(&self) -> AttentionTileMatmul {
        self.value_matmul
    }

    pub fn tile_size(&self) -> AttentionTileSize {
        self.tile_size
    }

    pub fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    pub fn num_planes(&self) -> u32 {
        self.num_planes
    }

    pub fn inner_layout(&self) -> InnerLayout {
        self.inner_layout
    }

    pub fn causal_mask(&self) -> bool {
        self.causal_mask
    }

    pub fn materialized_mask(&self) -> bool {
        self.materialized_mask
    }

    /// Workspace kind for the softmax round-trip, chosen by the score matmul
    /// variant.
    pub fn softmax_kind(&self) -> SoftmaxKind {
        match self.score_matmul {
            AttentionTileMatmul::Register(_) => SoftmaxKind::Direct {
                num_rows_per_unit: self.tile_size.seq_q,
            },
            AttentionTileMatmul::Cmma(_) => SoftmaxKind::Plane {
                inner_layout: self.inner_layout,
            },
        }
    }

    /// Bounce config for the score-tile round-trip (only meaningful for the
    /// cmma path).
    pub fn score_bounce_config(&self) -> BounceConfig {
        BounceConfig {
            tile_shape: (self.tile_size.seq_q, self.tile_size.seq_kv),
            num_planes: self.num_planes,
            plane_dim: self.plane_dim,
            inner_layout: self.inner_layout,
        }
    }

    /// Bounce config for the output (value-matmul accumulator) round-trip.
    pub fn output_bounce_config(&self) -> BounceConfig {
        BounceConfig {
            tile_shape: (self.tile_size.seq_q, self.tile_size.val_dim),
            num_planes: self.num_planes,
            plane_dim: self.plane_dim,
            inner_layout: self.inner_layout,
        }
    }

    /// Layout of the mask fragment, chosen by the score matmul variant.
    pub fn mask_layout(&self) -> MaskLayout {
        match self.score_matmul {
            AttentionTileMatmul::Register(_) => {
                MaskLayout::unit(self.tile_size.seq_q, self.tile_size.seq_kv)
            }
            AttentionTileMatmul::Cmma(_) => MaskLayout::local(
                (self.tile_size.seq_q, self.tile_size.seq_kv),
                self.plane_dim,
                self.inner_layout,
            ),
        }
    }
}

/// Selector for which tile-attention strategy to instantiate. Mirrors
/// matmul's `TileMatmulKind`: this is the surface used *before* the typed
/// configuration exists, owning availability checks and the
/// blueprint→config lowering.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileAttentionKind {
    /// Each unit independently runs a full register-based attention tile.
    Unit,
    /// Cmma-accelerated tile that round-trips fragments through smem for
    /// row-wise softmax/output operations.
    BlackboxAccelerated,
}

impl TileAttentionKind {
    /// Returns whether this tile attention requires specialized hardware
    /// accelerators (e.g. tensor cores).
    pub fn requires_accelerator(&self) -> bool {
        match self {
            TileAttentionKind::Unit => false,
            TileAttentionKind::BlackboxAccelerated => false,
        }
    }

    /// Returns the compute resources required.
    pub fn computation_resources(&self) -> Result<CubeDimResource, InvalidConfigError> {
        Ok(match self {
            TileAttentionKind::Unit => CubeDimResource::Units(1),
            TileAttentionKind::BlackboxAccelerated => CubeDimResource::Planes(1),
        })
    }

    /// Constructs the [`TileAttention`] config from the algorithm's blueprint.
    pub fn expand_tile_attention(
        &self,
        device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<TileAttention, AttentionSetupError> {
        let inner_layout = if blueprint.two_rows_in_array_tile {
            InnerLayout::SplitRows
        } else {
            InnerLayout::Contiguous
        };
        let plane_dim = blueprint.plane_dim;
        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let tile_size = blueprint.tiling_scheme.tile_size;

        let (score_matmul, value_matmul) = match self {
            TileAttentionKind::Unit => (
                AttentionTileMatmul::new_register_unit(tile_size.to_score_matmul_tile_size()),
                AttentionTileMatmul::new_register_unit(tile_size.to_value_matmul_tile_size()),
            ),
            TileAttentionKind::BlackboxAccelerated => (
                AttentionTileMatmul::new_cmma(tile_size.to_score_matmul_tile_size(), plane_dim),
                AttentionTileMatmul::new_cmma(tile_size.to_value_matmul_tile_size(), plane_dim),
            ),
        };

        let cfg = TileAttention {
            score_matmul,
            value_matmul,
            tile_size,
            plane_dim,
            num_planes,
            inner_layout,
            causal_mask: blueprint.causal,
            materialized_mask: blueprint.masked,
        };

        match self {
            TileAttentionKind::Unit => validate_unit(&cfg, &blueprint.vector_sizes)?,
            TileAttentionKind::BlackboxAccelerated => {
                validate_blackbox(device_props, &cfg, blueprint.vector_sizes.mask, dtypes)?
            }
        }

        Ok(cfg)
    }
}

fn validate_unit(
    cfg: &TileAttention,
    vector_sizes: &AttentionVectorSizes,
) -> Result<(), AttentionSetupError> {
    let tile_size = cfg.tile_size;
    let check_divisible =
        |dim: u32, vec_size: u32, name: &str, vec_name: &str| -> Result<(), AttentionSetupError> {
            if !dim.is_multiple_of(vec_size) {
                return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
                    "Tile's {} ({:?}) must be divisible by {} vector size ({:?})",
                    name, dim, vec_name, vec_size
                ))));
            }
            Ok(())
        };

    check_divisible(
        tile_size.head_dim,
        vector_sizes.query as u32,
        "head_dim",
        "query",
    )?;
    check_divisible(tile_size.seq_kv, vector_sizes.key as u32, "seq_kv", "key")?;
    check_divisible(
        tile_size.head_dim,
        vector_sizes.key as u32,
        "head_dim",
        "key",
    )?;
    check_divisible(tile_size.seq_kv, vector_sizes.mask as u32, "seq_kv", "mask")?;
    check_divisible(tile_size.val_dim, vector_sizes.out as u32, "val_dim", "out")?;
    check_divisible(
        tile_size.val_dim,
        vector_sizes.value as u32,
        "val_dim",
        "value",
    )?;
    Ok(())
}

fn validate_blackbox(
    device_props: &DeviceProperties,
    cfg: &TileAttention,
    line_sizes_mask: VectorSize,
    dtypes: &AttentionElems,
) -> Result<(), AttentionSetupError> {
    if dtypes.query_global != dtypes.query_tile {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Query global and tile types must be the same because no stage to cast in between",
        )));
    }

    if !device_props.features.matmul.cmma.contains(&MmaConfig {
        a_type: dtypes.query_tile,
        b_type: dtypes.key_value_tile,
        cd_type: dtypes.softmax_acc,
        m: cfg.tile_size.seq_q,
        k: cfg.tile_size.head_dim,
        n: cfg.tile_size.seq_kv,
    }) {
        return Err(AttentionSetupError::Unavailable(
            AttentionAvailabilityError::MatmulInstructionUnavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs: dtypes.query_tile,
                    rhs: dtypes.key_value_tile,
                    output: dtypes.softmax_acc,
                    size: Some(cfg.tile_size.to_score_matmul_tile_size()),
                },
            ),
        ));
    }
    if !device_props.features.matmul.cmma.contains(&MmaConfig {
        a_type: dtypes.softmax_lhs,
        b_type: dtypes.key_value_tile,
        cd_type: dtypes.accumulator,
        m: cfg.tile_size.seq_q,
        k: cfg.tile_size.seq_kv,
        n: cfg.tile_size.val_dim,
    }) {
        return Err(AttentionSetupError::Unavailable(
            AttentionAvailabilityError::MatmulInstructionUnavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs: dtypes.softmax_lhs,
                    rhs: dtypes.key_value_tile,
                    output: dtypes.accumulator,
                    size: Some(cfg.tile_size.to_value_matmul_tile_size()),
                },
            ),
        ));
    }

    if line_sizes_mask > 1 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Line size mask > 1 not supported yet on accelerated tile attention",
        )));
    }

    let softmax_num_rows = cfg.tile_size.seq_q;
    let softmax_num_cols = cfg.tile_size.seq_kv;
    let softmax_total = softmax_num_rows * softmax_num_cols;

    if !softmax_total.is_multiple_of(cfg.plane_dim) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Softmax size should be divisible by plane dim",
        )));
    }

    if cfg.inner_layout == InnerLayout::Contiguous && softmax_num_rows > cfg.plane_dim {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "More than one row per unit not supported with this inner layout",
        )));
    }

    if cfg.inner_layout == InnerLayout::SplitRows
        && !softmax_total.is_multiple_of(2 * cfg.plane_dim)
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "With split rows, units must have two elements each",
        )));
    }

    if cfg.tile_size.head_dim < cfg.tile_size.val_dim {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Can't have tile head_dim < tile val dim (not sure why)",
        )));
    }

    Ok(())
}
