use cubecl::CubeDim;

use crate::{
    components::global::multi_stage::LoadMaxRoundPlaneCount,
    definition::{MatmulElems, MatmulLineSizes, MatmulSetupError, TilingScheme},
};

#[allow(unused_variables)]
pub fn cube_dim_validation(cube_dim: CubeDim) -> Result<(), MatmulSetupError> {
    #[cfg(target_os = "macos")]
    {
        if cube_dim.num_elems() >= 512 {
            use crate::definition::{MatmulAvailabilityError, MatmulSetupError};

            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CubeDimTooBig(cube_dim),
            ));
        }
    }

    Ok(())
}

/// Maximal number of planes each reader can handle to divide its workload evenly
pub struct MaxGlobalReaderPlanes {
    pub lhs: u32,
    pub rhs: u32,
}

impl MaxGlobalReaderPlanes {
    /// Create a MaxGlobalReaderPlanes
    pub fn new<LL: LoadMaxRoundPlaneCount, RL: LoadMaxRoundPlaneCount>(
        tiling_scheme: &TilingScheme,
        line_sizes: &MatmulLineSizes,
        plane_dim: u32,
        dtypes: &MatmulElems,
    ) -> Self {
        MaxGlobalReaderPlanes {
            lhs: LL::max_round_plane_count(
                tiling_scheme.tile_size.m * tiling_scheme.tile_size.k,
                (tiling_scheme.partition_size.m
                    * tiling_scheme.stage_size.m
                    * tiling_scheme.partition_size.k
                    * tiling_scheme.stage_size.k) as u32,
                line_sizes.lhs,
                plane_dim,
                dtypes.lhs_global,
            ),
            rhs: RL::max_round_plane_count(
                tiling_scheme.tile_size.k * tiling_scheme.tile_size.n,
                (tiling_scheme.partition_size.k
                    * tiling_scheme.stage_size.k
                    * tiling_scheme.partition_size.n
                    * tiling_scheme.stage_size.n) as u32,
                line_sizes.rhs,
                plane_dim,
                dtypes.rhs_global,
            ),
        }
    }
}
