use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::components::batch::partitioned_matmul::partition::{
    GlobalPartitionMatmul, PartitionRangeDim, PartitionRanges,
};
use crate::definition::{
    AccG, Blueprint as _, CubeMapping, LhsG, MatmulElems, MatmulTypes, MatmulVectorSizes, RhsG,
    TilingBlueprint, cube_pos_to_m_n_batch,
};
use crate::launch::MatmulArgs;
use crate::{
    components::batch::partitioned_matmul::config::PartitionedBatchConfig, launch::RuntimeConfig,
};
use crate::{
    components::batch::{BatchMatmul, BatchMatmulFamily, PartitionedBatchMatmulFamily},
    components::global::{self, GlobalConfig, GlobalMatmul, GlobalMatmulFamily},
    components::stage::StageConfig as _,
};

#[cube(launch_unchecked, explicit_define, address_type = "dynamic")]
#[allow(clippy::type_complexity)]
/// Launches the matmul kernel
pub(crate) fn matmul_entry<
    Args: MatmulArgs,
    Lhs: Numeric,
    LhsSize: Size,
    Rhs: Numeric,
    RhsSize: Size,
    Acc: Numeric,
    AccSize: Size,
    GMMF: GlobalMatmulFamily<Args::Config>,
    GPM: GlobalPartitionMatmul,
>(
    inputs: &<Args as MatmulArgs>::Input<
        Vector<Lhs, LhsSize>,
        Vector<Rhs, RhsSize>,
        Vector<Acc, AccSize>,
    >,
    output: &mut <Args as MatmulArgs>::Output<Vector<Acc, AccSize>>,
    config: <Args as MatmulArgs>::Config,
    cube_mapping: CubeMapping,
    #[comptime] blueprint: TilingBlueprint,
    #[comptime] dtypes: MatmulElems,
    #[define(Lhs, Rhs, Acc)] _global: [StorageType; 3],
    #[define(LhsSize, RhsSize, AccSize)] _sizes: [usize; 3],
) {
    let mut state =
        Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
            inputs,
            output,
            config,
            blueprint.lhs_global_layout_config(),
            blueprint.rhs_global_layout_config(),
            blueprint.out_global_layout_config(),
        );

    let vector_size_lhs = Args::view_lhs(&state).vector_size();
    let vector_size_rhs = Args::view_rhs(&state).vector_size();
    let vector_size_out = Args::view_out(&mut state).vector_size();
    let vector_sizes = comptime!(MatmulVectorSizes {
        lhs: vector_size_lhs,
        rhs: vector_size_rhs,
        out: vector_size_out,
    });

    let device_props = comptime::device_properties();
    let config = comptime!(
        PartitionedBatchMatmulFamily::<Args::Config, GMMF, GPM>::expand_config(
            &device_props,
            &blueprint,
            &dtypes,
            &vector_sizes
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

    let stage_lhs = config.global_config.stage_config().lhs_smem_config();
    let stage_rhs = config.global_config.stage_config().rhs_smem_config();
    let stage_acc = config.global_config.stage_config().acc_smem_config();

    let define!(StageLhs) = stage_lhs.dtype;
    let size!(StageLhsSize) = comptime![stage_lhs.vector_size as usize];
    let define!(RegisterLhs) = dtypes.lhs_register;

    let define!(StageRhs) = stage_rhs.dtype;
    let size!(StageRhsSize) = comptime![stage_rhs.vector_size as usize];
    let define!(RegisterRhs) = dtypes.rhs_register;

    let define!(StageAcc) = stage_acc.dtype;
    let size!(StageAccSize) = comptime![stage_acc.vector_size as usize];
    let define!(RegisterAcc) = dtypes.acc_register;

    PartitionedBatchMatmul::<
        Args::Config,
        (
            (
                Lhs,
                LhsSize,
                StageLhs,
                StageLhsSize,
                RegisterLhs,
                StageLhsSize,
            ),
            (
                Rhs,
                RhsSize,
                StageRhs,
                StageRhsSize,
                RegisterRhs,
                StageRhsSize,
            ),
            (
                Acc,
                AccSize,
                StageAcc,
                StageAccSize,
                RegisterAcc,
                StageAccSize,
            ),
        ),
        GMMF::Matmul<(
            (
                Lhs,
                LhsSize,
                StageLhs,
                StageLhsSize,
                RegisterLhs,
                StageLhsSize,
            ),
            (
                Rhs,
                RhsSize,
                StageRhs,
                StageRhsSize,
                RegisterRhs,
                StageRhsSize,
            ),
            (
                Acc,
                AccSize,
                StageAcc,
                StageAccSize,
                RegisterAcc,
                StageAccSize,
            ),
        )>,
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
    MP: MatmulTypes,
    GMM: global::GlobalMatmul<RC, MP>,
    S: GlobalPartitionMatmul,
> {
    _rc: PhantomData<RC>,
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

#[cube]
impl<RC: RuntimeConfig, MP: MatmulTypes, GMM: GlobalMatmul<RC, MP>, GPMM: GlobalPartitionMatmul>
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

        let (m_index, n_index, batch_index) = cube_pos_to_m_n_batch(&cube_mapping);

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
