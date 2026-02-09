use crate::components::{
    global::{
        GlobalMatmul, GlobalWriter, SharedGlobalMatmulConfig,
        read::{FullLoaderStage, FullLoadingStrategy, FullStageGlobalReader, SyncStrategy},
    },
    stage::{StageConfig, StageMatmul},
    tile::io::Strided,
};
use crate::{
    definition::{AccG, AccS, LhsG, LhsS, MatmulPrecision, MatrixPrecision, RhsG, RhsS},
    launch::RuntimeConfig,
};
use cubecl::prelude::*;
use cubecl::std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level.
///
/// Fully loads all stages, synchronizes all planes, performs computation,
/// synchronizes again, then proceeds to the next set of stages.
pub struct SimpleMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
> {
    _phantom: PhantomData<(MP, SMM, RC, LL, RL, AL, GW)>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, RC, LL, RL, AL, GW> GlobalMatmul<RC, MP>
    for SimpleMatmul<MP, SMM, RC, LL, RL, AL, GW>
where
    SMM: StageMatmul<
            MP,
            LhsStage = FullLoaderStage<RC, LL, LhsS<MP>>,
            RhsStage = FullLoaderStage<RC, RL, RhsS<MP>>,
            AccStage = CubeOption<FullLoaderStage<RC, AL, AccS<MP>>>,
            OutStage = GW::Stage,
        >,
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC, TileKind = Strided>,
    RL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, TileKind = Strided>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig<SMM::Config>;
    type LhsGlobalReader = FullStageGlobalReader<
        <MP::Lhs as MatrixPrecision>::Global,
        <MP::Lhs as MatrixPrecision>::Stage,
        RC,
        LL,
    >;
    type RhsGlobalReader = FullStageGlobalReader<
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

        let k_step = config.stage_config.elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let mut acc = SMM::init_accumulators(config.stage_config);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config);

        let mut barrier = LL::SyncStrategy::create_barrier();

        let acc_stage = match acc_reader {
            CubeOption::Some(mut reader) => {
                let mut acc_barrier = AL::SyncStrategy::create_barrier();
                reader.load_stage(&mut acc_barrier, config.acc_reader_config);
                AL::SyncStrategy::sync::<MP, _>(&mut acc_barrier, config);
                CubeOption::new_Some(reader.stage())
            }
            CubeOption::None => CubeOption::new_None(),
        };

        SMM::load_accumulators(&acc_stage, &mut acc, config.stage_config);

        let lhs_stage = &lhs_reader.stage();
        let rhs_stage = &rhs_reader.stage();

        for _ in 0..num_loops {
            sync_cube();

            lhs_reader.load_stage(&mut barrier, config.lhs_reader_config);
            rhs_reader.load_stage(&mut barrier, config.rhs_reader_config);

            LL::SyncStrategy::sync::<MP, _>(&mut barrier, config);

            SMM::execute(
                lhs_stage,
                rhs_stage,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                config.stage_config,
                &partition_scheduler,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();
        }

        // Frees input stages for reuse, so the output stage can be allocated into the same
        // range. The `sync_cube` is required to ensure other planes are done reading from the stages.
        //
        // This is currently very unintuitive, because while the stage already exists, it actually
        // isn't allocated until it's used (by writing to it). We should eventually separate the
        // write call into a different function and defer creating the writer until after the stages
        // are freed to make the order of operations more clear.
        sync_cube();
        lhs_reader.free_stage();
        rhs_reader.free_stage();

        let mut out_stage = Self::GlobalWriter::stage(&out_writer);

        SMM::write_results::<Self::GlobalWriter>(
            &mut acc,
            &mut out_stage,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config,
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        Self::LhsGlobalReader::new(
            lhs,
            runtime_config,
            config.stage_config.elements_in_stage_k(),
            config.lhs_reader_config,
        )
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        Self::RhsGlobalReader::new(
            rhs,
            runtime_config,
            config.stage_config.elements_in_stage_k(),
            config.rhs_reader_config,
        )
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
