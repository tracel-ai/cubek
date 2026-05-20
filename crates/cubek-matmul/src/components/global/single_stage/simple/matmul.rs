use crate::launch::RuntimeConfig;
use crate::{
    components::{
        global::{
            GlobalMatmul, GlobalWriter, SharedGlobalMatmulConfig,
            read::{FullLoaderStage, FullLoadingStrategy, FullStageGlobalReader, SyncStrategy},
        },
        stage::{
            {StagePartitioner, partition_coordinates},
            {init_a_fragment, init_accumulator, init_b_fragments},
        },
    },
    definition::*,
};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d},
};
use cubek_std::tile::{
    NoEvent, PartitionScheduler, Tile, load_partition_from_stage, write_partition_to_stage,
};
use std::marker::PhantomData;

// Type aliases for the (long) per-flow Stage types — saves repeating the
// `FullLoaderStage<RC, LL, Stage<Lhs<MP>>, StageSize<Lhs<MP>>>` shape at every
// PartitionMatmul call site below.
type LhsStageFor<MP, RC, LL> = FullLoaderStage<RC, LL, Stage<Lhs<MP>>, StageSize<Lhs<MP>>>;
type RhsStageFor<MP, RC, RL> = FullLoaderStage<RC, RL, Stage<Rhs<MP>>, StageSize<Rhs<MP>>>;
type AccStageFor<MP, RC, AL> =
    ComptimeOption<FullLoaderStage<RC, AL, Stage<Acc<MP>>, StageSize<Acc<MP>>>>;

/// Performs matrix multiplication at the global level.
///
/// Fully loads all stages, synchronizes all planes, performs computation,
/// synchronizes again, then proceeds to the next set of stages.
pub struct SimpleMatmul<
    MP: MatmulTypes,
    SP: StagePartitioner,
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
> {
    _phantom: PhantomData<(MP, SP, RC, LL, RL, AL, GW)>,
}

#[cube]
impl<MP: MatmulTypes, SP, RC, LL, RL, AL, GW> GlobalMatmul<RC, MP>
    for SimpleMatmul<MP, SP, RC, LL, RL, AL, GW>
where
    SP: StagePartitioner,
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig;
    type LhsGlobalReader = FullStageGlobalReader<
        <MP::Lhs as MatrixTypes>::Global,
        <MP::Lhs as MatrixTypes>::GlobalSize,
        <MP::Lhs as MatrixTypes>::Stage,
        <MP::Lhs as MatrixTypes>::StageSize,
        RC,
        LL,
    >;
    type RhsGlobalReader = FullStageGlobalReader<
        <MP::Rhs as MatrixTypes>::Global,
        <MP::Rhs as MatrixTypes>::GlobalSize,
        <MP::Rhs as MatrixTypes>::Stage,
        <MP::Rhs as MatrixTypes>::StageSize,
        RC,
        RL,
    >;
    type AccGlobalReader = ComptimeOption<
        FullStageGlobalReader<
            <MP::Acc as MatrixTypes>::Global,
            <MP::Acc as MatrixTypes>::GlobalSize,
            <MP::Acc as MatrixTypes>::Stage,
            <MP::Acc as MatrixTypes>::StageSize,
            RC,
            AL,
        >,
    >;
    type GlobalWriter = GW;
    type Accumulators = Tile<AccRE<MP>, SP::Scope>;

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

        let stage_shared = config.stage_config.shared();

        let mut acc = init_accumulator::<MP, SP::Scope>(stage_shared);
        let mut a_fragment = init_a_fragment::<MP, SP::Scope>(stage_shared);

        let (partition_row, partition_col) = partition_coordinates::<SP>(
            stage_shared.plane_flow_config.partition_rule,
            stage_shared.plane_dim,
            stage_shared.stage_size.n(),
        );
        let partition_scheduler = PartitionScheduler::new(
            partition_row,
            partition_col,
            stage_shared.partition_size,
            stage_shared.partition_schedule_scheme,
        );

        let mut barrier = LL::SyncStrategy::create_barrier();

        let acc_stage = acc_reader.map(|mut reader| {
            let mut acc_barrier = AL::SyncStrategy::create_barrier();
            reader.load_stage(&mut acc_barrier, config.acc_reader_config);
            AL::SyncStrategy::sync::<MP>(&mut acc_barrier, config);
            reader.stage()
        });
        load_partition_from_stage::<
            AccSE<MP>,
            AccSS<MP>,
            LhsRE<MP>,
            RhsRE<MP>,
            AccRE<MP>,
            SP::Scope,
            AccStageFor<MP, RC, AL>,
        >(
            &acc_stage,
            &mut acc,
            &partition_scheduler,
            stage_shared.partition_size.m(),
            stage_shared.partition_size.n(),
        );

        let lhs_stage = lhs_reader.stage();
        let rhs_stage = rhs_reader.stage();
        let lhs_stage_tile = <LhsStageFor<MP, RC, LL> as crate::components::stage::Stage<
            Stage<Lhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&lhs_stage);
        let rhs_stage_tile = <RhsStageFor<MP, RC, RL> as crate::components::stage::Stage<
            Stage<Rhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&rhs_stage);

        let mut b_fragments = init_b_fragments::<MP, SP::Scope>(stage_shared);

        for _ in 0..num_loops {
            sync_cube();
            lhs_reader.load_stage(&mut barrier, config.lhs_reader_config);
            rhs_reader.load_stage(&mut barrier, config.rhs_reader_config);
            LL::SyncStrategy::sync::<MP>(&mut barrier, config);
            acc.mma_partition::<
                LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                NoEvent,
            >(
                &lhs_stage_tile,
                &rhs_stage_tile,
                &mut a_fragment,
                &mut b_fragments,
                stage_shared.partition_size.k(),
                NoEvent::new(),
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

        write_partition_to_stage::<
            <MP::Acc as MatrixTypes>::Stage,
            AccSS<MP>,
            LhsRE<MP>,
            RhsRE<MP>,
            AccRE<MP>,
            SP::Scope,
            GW::Stage,
            GW,
        >(
            &mut acc,
            &mut out_stage,
            &mut out_writer,
            &partition_scheduler,
            stage_shared.partition_size.m(),
            stage_shared.partition_size.n(),
        );
    }

    fn init_lhs_global_reader(
        lhs: View<LhsG<MP>, Coords2d>,
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
        rhs: View<RhsG<MP>, Coords2d>,
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
        acc: ComptimeOption<View<AccG<MP>, Coords2d>>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader {
        acc.map(|view| {
            FullStageGlobalReader::new(view, runtime_config, 0, config.acc_reader_config)
        })
    }

    fn init_global_writer(
        out: View<AccG<MP>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        Self::GlobalWriter::init(out, config.writer_config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        init_accumulator::<MP, SP::Scope>(config.stage_config.shared())
    }
}
