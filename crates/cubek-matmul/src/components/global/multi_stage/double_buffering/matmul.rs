use crate::components::global::{
    GlobalMatmul, GlobalWriter, SharedGlobalMatmulConfig,
    read::{FullLoaderStage, PartialLoaderStage},
};
use crate::components::global::{Specializer, read::SyncStrategy};
use crate::components::global::{
    multi_stage::double_buffer_execution::{
        execute_current_and_read_next, execute_last_and_write_results, read_first,
    },
    read::FullLoadingStrategy,
};
use crate::components::stage;
use crate::components::stage::StageConfig;
use crate::components::{
    global::read::{
        FullStageGlobalReader, PartialLoadingStrategy, PartialStageGlobalReader, StageBuffer,
    },
    tile::io::Strided,
};
use crate::definition::{AccG, AccS, LhsG, LhsS, MatmulPrecision, MatrixPrecision, RhsG, RhsS};
use crate::launch::RuntimeConfig;
use cubecl::prelude::*;
use cubecl::std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on stage A,
/// they trigger a computation event from tensor cores on stage B. Then stages are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    RC: RuntimeConfig,
    LL: PartialLoadingStrategy<RC>,
    RL: PartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _rc: PhantomData<RC>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, RC, LL, RL, AL, GW> GlobalMatmul<RC, MP>
    for DoubleBufferingMatmul<MP, SMM, RC, LL, RL, AL, GW>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStage = PartialLoaderStage<RC, LL, LhsS<MP>>,
            RhsStage = PartialLoaderStage<RC, RL, RhsS<MP>>,
            AccStage = CubeOption<FullLoaderStage<RC, AL, AccS<MP>>>,
            OutStage = GW::Stage,
        >,
    RC: RuntimeConfig,
    LL: PartialLoadingStrategy<RC, TileKind = Strided>,
    RL: PartialLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    type LhsGlobalReader = PartialStageGlobalReader<
        <MP::Lhs as MatrixPrecision>::Global,
        <MP::Lhs as MatrixPrecision>::Stage,
        RC,
        LL,
    >;
    type RhsGlobalReader = PartialStageGlobalReader<
        <MP::Rhs as MatrixPrecision>::Global,
        <MP::Rhs as MatrixPrecision>::Stage,
        RC,
        RL,
    >;
    type AccGlobalReader = CubeOption<
        FullStageGlobalReader<
            <MP::Acc as MatrixPrecision>::Global,
            <MP::Acc as MatrixPrecision>::Stage,
            RC,
            AL,
        >,
    >;

    type GlobalWriter = GW;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_reader: Self::LhsGlobalReader,
        mut rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        mut out_writer: Self::GlobalWriter,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let device_props = comptime::device_properties();
        if let Err(e) = comptime!(LL::validate_with_config(
            &device_props,
            &config.lhs_reader_config
        )) {
            push_validation_error(e.to_string());
            comptime!(return);
        }
        if let Err(e) = comptime!(RL::validate_with_config(
            &device_props,
            &config.rhs_reader_config
        )) {
            push_validation_error(e.to_string());
            comptime!(return);
        }

        let stage_step = config.stage_config.elements_in_stage_k();

        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = range.div_ceil(stage_step);

        let mut acc = SMM::init_accumulators(config.stage_config);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config);

        let lhs_stage_a = lhs_reader.stage(StageBuffer::A);
        let lhs_stage_b = lhs_reader.stage(StageBuffer::B);
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let mut barrier_a = LL::SyncStrategy::create_barrier();
        let mut barrier_b = LL::SyncStrategy::create_barrier();

        let specializer = Specializer::new(
            config.plane_flow_config(),
            config.specialized_loading_sides(),
        );

        let acc_stage = match acc_reader {
            CubeOption::Some(mut reader) => {
                reader.load_stage(&mut barrier_a, config.acc_reader_config);
                LL::SyncStrategy::sync::<MP, _>(&mut barrier_a, config);
                CubeOption::new_Some(reader.stage())
            }
            CubeOption::None => CubeOption::new_None(),
        };

        SMM::load_accumulators(&acc_stage, &mut acc, config.stage_config);

        read_first::<LL::SyncStrategy, Self::LhsGlobalReader, Self::RhsGlobalReader>(
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier_a,
            &specializer,
            StageBuffer::A,
            config.lhs_reader_config,
            config.rhs_reader_config,
        );

        LL::SyncStrategy::sync::<MP, _>(&mut barrier_a, config);

        for _ in 0..num_loops {
            execute_current_and_read_next::<
                MP,
                SMM,
                LL::SyncStrategy,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage_a,
                &rhs_stage_a,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier_b,
                &specializer,
                &partition_scheduler,
                StageBuffer::B,
                config,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();

            LL::SyncStrategy::sync::<MP, _>(&mut barrier_b, config);

            execute_current_and_read_next::<
                MP,
                SMM,
                LL::SyncStrategy,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage_b,
                &rhs_stage_b,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier_a,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            LL::SyncStrategy::sync::<MP, _>(&mut barrier_a, config);
        }

        execute_current_and_read_next::<
            MP,
            SMM,
            LL::SyncStrategy,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &lhs_stage_a,
            &rhs_stage_a,
            &mut lhs_tile,
            &mut rhs_tile,
            &mut acc,
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier_b,
            &specializer,
            &partition_scheduler,
            StageBuffer::B,
            config,
        );

        LL::SyncStrategy::sync::<MP, _>(&mut barrier_b, config);

        execute_last_and_write_results::<MP, GW, SMM, Self::Config>(
            &lhs_stage_b,
            &rhs_stage_b,
            &mut lhs_tile,
            &mut rhs_tile,
            &mut acc,
            &mut out_writer,
            &specializer,
            &partition_scheduler,
            config,
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        // We always advance by 2 * k because stage B shares the same global memory state as stage A,
        // but it is implicitly offset by one stage's worth (k elements) when reading.
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::<
            <MP::Lhs as MatrixPrecision>::Global,
            <MP::Lhs as MatrixPrecision>::Stage,
            RC,
            LL,
        >::new(lhs, runtime_config, k_step, config.lhs_reader_config)
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        // We always advance by 2 * k because stage B shares the same global memory state as stage A,
        // but it is implicitly offset by one stage's worth (k elements) when reading.
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::<
            <MP::Rhs as MatrixPrecision>::Global,
            <MP::Rhs as MatrixPrecision>::Stage,
            RC,
            RL,
        >::new(rhs, runtime_config, k_step, config.rhs_reader_config)
    }

    fn init_acc_global_reader(
        acc: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader {
        match acc {
            CubeOption::None => CubeOption::new_None(),
            CubeOption::Some(view) => CubeOption::new_Some(FullStageGlobalReader::new(
                view,
                runtime_config,
                0,
                config.acc_reader_config,
            )),
        }
    }

    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        Self::GlobalWriter::init(out, config.writer_config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config)
    }
}
