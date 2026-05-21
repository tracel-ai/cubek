use crate::components::global::{Specializer, read::sync::Synchronous};
use crate::components::{
    global::multi_stage::double_buffer_execution::{
        execute_current_and_read_next, execute_last_and_write_results, read_first,
    },
    stage::{
        {StagePartitioner, partition_coordinates},
        {init_a_fragment, init_accumulator, init_b_fragments},
    },
};
use crate::{components::global::multi_stage::ordered::LL, launch::RuntimeConfig};
use crate::{
    components::global::read::{
        FullLoaderStage, FullLoadingStrategy, FullStageGlobalReader, LoadingValidation as _,
        PartialLoaderStage, PartialLoadingStrategy, PartialStageGlobalReader, StageBuffer,
    },
    definition::{Rhs, Stage},
};
use crate::{
    components::global::{self, GlobalWriter, SharedGlobalMatmulConfig},
    definition::*,
};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d},
};
use cubek_std::tile::{PartitionScheduler, Tile, load_partition_from_stage};
use std::marker::PhantomData;

// Per-flow Stage type aliases — keep call sites readable.
type LhsStageFor<MP, RC> = FullLoaderStage<RC, LL, Stage<Lhs<MP>>, StageSize<Lhs<MP>>>;
type RhsStageFor<MP, RC, RL> = PartialLoaderStage<RC, RL, Stage<Rhs<MP>>, StageSize<Rhs<MP>>>;
type AccStageFor<MP, RC, AL> =
    ComptimeOption<FullLoaderStage<RC, AL, Stage<Acc<MP>>, StageSize<Acc<MP>>>>;

/// Performs matrix multiplication at the global level.
/// Uses double buffering with two shared memory buffers for `Rhs`,
/// but only one for `Lhs`—the second "buffer" for `Lhs` is the fragments themselves.
/// For this to work, the `Lhs` reader planes must compute using
/// only the data they have loaded themselves.
pub struct OrderedDoubleBufferingMatmul<
    MP: MatmulTypes,
    SP: StagePartitioner,
    RC: RuntimeConfig,
    RL: PartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
> {
    _ms: PhantomData<MP>,
    _sp: PhantomData<SP>,
    _rc: PhantomData<RC>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

#[cube]
impl<MP: MatmulTypes, SP, RC, RL, AL, GW> global::GlobalMatmul<RC, MP>
    for OrderedDoubleBufferingMatmul<MP, SP, RC, RL, AL, GW>
where
    SP: StagePartitioner,
    RC: RuntimeConfig,
    RL: PartialLoadingStrategy<RC, SyncStrategy = Synchronous>,
    AL: FullLoadingStrategy<RC, SyncStrategy = Synchronous>,
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
    type RhsGlobalReader = PartialStageGlobalReader<
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
        if config
            .lhs_reader_config
            .input_load_flow
            .has_specialization()
        {
            push_validation_error(
                "Error: In Ordered lhs loading cannot be outside of main flow".to_string(),
            );
            comptime!(return);
        }

        let stage_step = config.stage_config.elements_in_stage_k();

        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = range.div_ceil(stage_step);

        let stage_shared = config.stage_config.shared();

        let mut acc = init_accumulator::<MP, SP::Scope>(stage_shared);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        let mut barrier = ();

        let acc_stage = acc_reader.map(|mut reader| {
            reader.load_stage(&mut barrier, config.acc_reader_config);
            sync_cube();
            reader.stage()
        });

        let mut lhs_tile = init_a_fragment::<MP, SP::Scope>(stage_shared);
        let mut rhs_tile = init_b_fragments::<MP, SP::Scope>(stage_shared);

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
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);
        let lhs_stage_tile = <LhsStageFor<MP, RC> as crate::components::stage::Stage<
            Stage<Lhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&lhs_stage);
        let rhs_stage_a_tile = <RhsStageFor<MP, RC, RL> as crate::components::stage::Stage<
            Stage<Rhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&rhs_stage_a);
        let rhs_stage_b_tile = <RhsStageFor<MP, RC, RL> as crate::components::stage::Stage<
            Stage<Rhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&rhs_stage_b);

        let specializer = Specializer::new(
            config.plane_flow_config(),
            config.specialized_loading_sides(),
        );

        read_first::<Synchronous, Self::LhsGlobalReader, Self::RhsGlobalReader>(
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier,
            &specializer,
            StageBuffer::A,
            config.lhs_reader_config,
            config.rhs_reader_config,
        );

        lhs_reader.advance_view();

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_read_next::<
                MP,
                SP,
                Synchronous,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage_tile,
                &rhs_stage_a_tile,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier,
                &specializer,
                &partition_scheduler,
                StageBuffer::B,
                config,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();

            sync_cube();

            execute_current_and_read_next::<
                MP,
                SP,
                Synchronous,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage_tile,
                &rhs_stage_b_tile,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            lhs_reader.advance_view();

            sync_cube();
        }

        execute_current_and_read_next::<
            MP,
            SP,
            Synchronous,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &lhs_stage_tile,
            &rhs_stage_a_tile,
            &mut lhs_tile,
            &mut rhs_tile,
            &mut acc,
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier,
            &specializer,
            &partition_scheduler,
            StageBuffer::B,
            config,
        );

        sync_cube();

        execute_last_and_write_results::<MP, GW, SP, Self::Config>(
            &lhs_stage_tile,
            &rhs_stage_b_tile,
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
        lhs: View<LhsG<MP>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        // We always advance by only k for Lhs
        let k_step = config.stage_config.elements_in_stage_k();
        FullStageGlobalReader::new(lhs, runtime_config, k_step, config.lhs_reader_config)
    }

    fn init_rhs_global_reader(
        rhs: View<RhsG<MP>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        // We always advance by 2 * k for Rhs only
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::new(rhs, runtime_config, k_step, config.rhs_reader_config)
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
