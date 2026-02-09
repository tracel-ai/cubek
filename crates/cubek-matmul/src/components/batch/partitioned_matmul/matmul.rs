use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::components::batch::partitioned_matmul::partition::{
    GlobalPartitionMatmul, PartitionRangeDim, PartitionRanges,
};
use crate::components::batch::{BatchMatmul, BatchMatmulFamily, PartitionedBatchMatmulFamily};
use crate::components::global::{self, GlobalConfig, GlobalMatmul, GlobalMatmulFamily};
use crate::components::stage::StageConfig as _;
use crate::definition::{
    AccG, Blueprint as _, CubeMapping, LhsG, MatmulElems, MatmulLineSizes, MatmulPrecision, RhsG,
    TilingBlueprint,
};
use crate::launch::MatmulArgs;
use crate::{
    components::batch::partitioned_matmul::config::PartitionedBatchConfig, launch::RuntimeConfig,
};

#[cube(launch_unchecked)]
/// Launches the matmul kernel
pub(crate) fn matmul_entry<
    Args: MatmulArgs,
    LhsG: Numeric,
    RhsG: Numeric,
    AccG: Numeric,
    LhsS: Numeric,
    RhsS: Numeric,
    AccS: Numeric,
    LhsR: Numeric,
    RhsR: Numeric,
    AccR: Numeric,
    GMMF: GlobalMatmulFamily<Args::Config>,
    GPM: GlobalPartitionMatmul,
>(
    inputs: &<Args as MatmulArgs>::Input<LhsG, RhsG, AccG>,
    output: &mut <Args as MatmulArgs>::Output<AccG>,
    config: <Args as MatmulArgs>::Config,
    cube_mapping: CubeMapping,
    #[comptime] blueprint: TilingBlueprint,
    #[define(LhsG, RhsG, AccG)] global: [StorageType; 3],
    #[define(LhsS, RhsS, AccS)] stage: [StorageType; 3],
    #[define(LhsR, RhsR, AccR)] register: [StorageType; 3],
) {
    let mut state = Args::init_state::<LhsG, RhsG, AccG>(
        inputs,
        output,
        config,
        blueprint.lhs_global_layout_config(),
        blueprint.rhs_global_layout_config(),
        blueprint.out_global_layout_config(),
    );

    let line_size_lhs = Args::view_lhs(&state).line_size();
    let line_size_rhs = Args::view_rhs(&state).line_size();
    let line_size_out = Args::view_out(&mut state).line_size();
    let line_sizes = comptime!(MatmulLineSizes {
        lhs: line_size_lhs,
        rhs: line_size_rhs,
        out: line_size_out,
    });

    let device_props = comptime::device_properties();
    let config = comptime!(
        PartitionedBatchMatmulFamily::<Args::Config, GMMF, GPM>::expand_config(
            &device_props,
            &blueprint,
            &MatmulElems::from_define_arrays(global, stage, register),
            &line_sizes
        )
    );

    if comptime!(config.is_err()) {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }

    let config = comptime!(config.unwrap());

    #[allow(clippy::collapsible_if)]
    if cube_mapping.can_yield_extra_cubes {
        if CUBE_POS >= cube_mapping.num_valid_cubes() {
            terminate!()
        }
    }

    PartitionedBatchMatmul::<
        Args::Config,
        ((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR)),
        GMMF::Matmul<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>,
        GPM,
    >::execute::<Args>(&mut state, cube_mapping, config);
}

/// Executes matrix multiplication at the batch level,
/// assigning each cube to handle multiple global matmuls.
///
/// Each cube performs a number of global matmuls specified by
/// the global partition size of the tiling scheme
pub struct PartitionedBatchMatmul<
    RC: RuntimeConfig,
    MP: MatmulPrecision,
    GMM: global::GlobalMatmul<RC, MP>,
    S: GlobalPartitionMatmul,
> {
    _rc: PhantomData<RC>,
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

#[cube]
impl<RC: RuntimeConfig, MP: MatmulPrecision, GMM: GlobalMatmul<RC, MP>, GPMM: GlobalPartitionMatmul>
    BatchMatmul<RC, MP> for PartitionedBatchMatmul<RC, MP, GMM, GPMM>
{
    type Config = PartitionedBatchConfig<GMM::Config>;

    fn execute<Args: MatmulArgs<Config = RC>>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let (_, _, problem_k) = Args::view_lhs(state).shape();
        let k_range = (0, problem_k);

        let (m_index, n_index, batch_index) = cube_mapping.cube_pos_to_tensor_pos();

        let ranges = PartitionRanges::new(
            PartitionRangeDim::new(
                m_index,
                config.global_config.stage_config().elements_in_stage_m(),
                config.global_partition_size.m,
            ),
            PartitionRangeDim::new(
                n_index,
                config.global_config.stage_config().elements_in_stage_n(),
                config.global_partition_size.n,
            ),
            PartitionRangeDim::new(batch_index, 1u32, config.global_partition_size.batches),
        );

        GPMM::execute::<Args, MP, GMM>(state, ranges, k_range, config.global_config);
    }
}
