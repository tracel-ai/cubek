use cubecl;
use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::VirtualTensor;
use cubek_matmul::components::global::PartitionedStage;
use cubek_matmul::components::global::read::FullStageGlobalReader;
use cubek_matmul::components::stage::StridedStageMemory;
use std::marker::PhantomData;

use crate::components::global::AttentionGlobalLayout;
use crate::components::global::simple::QueryReader;
use crate::components::global::simple::{AttentionWriter, AttentionWriterExpand, MaskReader};
use crate::components::global::{GlobalAttention, simple::config::SimpleGlobalAttentionConfig};
use crate::components::stage::{
    AttentionLoadingStrategy, AttentionPartitioner, AttentionTilingLayout, StageAttention,
    StageAttentionConfig as _,
};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

pub struct SimpleGlobalAttention<AP: AttentionPrecision, SA: StageAttention<AP>> {
    _phantom: PhantomData<(AP, SA)>,
}

#[cube]
impl<
    SA: StageAttention<
            AP,
            KeyStage = StridedStageMemory<KS<AP>, KSS<AP>, AttentionTilingLayout>,
            ValueStage = StridedStageMemory<VS<AP>, VSS<AP>, AttentionTilingLayout>,
            OutStage = PartitionedStage<OS<AP>, OSS<AP>>,
        >,
    AP: AttentionPrecision,
> GlobalAttention<AP> for SimpleGlobalAttention<AP, SA>
{
    type KeyReader =
        FullStageGlobalReader<KG<AP>, KGS<AP>, KS<AP>, KSS<AP>, (), AttentionLoadingStrategy>;
    type ValueReader =
        FullStageGlobalReader<VG<AP>, VGS<AP>, VS<AP>, VSS<AP>, (), AttentionLoadingStrategy>;
    type MaskReader = MaskReader<AP>;

    type Writer =
        <SA::Partitioner as AttentionPartitioner>::Writer<OS<AP>, OSS<AP>, OG<AP>, OGS<AP>>;

    type Config = SimpleGlobalAttentionConfig<SA::Config>;

    fn execute(
        query_reader: QueryReader<AP>,
        mut key_reader: Self::KeyReader,
        mut value_reader: Self::ValueReader,
        mut mask_reader: Self::MaskReader,
        mut writer: Self::Writer,
        seq_q: u32,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    ) {
        // Load queries which stay alive in registers for all the kernel
        let mut query_registers = SA::init_query(config.stage_config);
        SA::read_query(&query_reader, &mut query_registers, config.stage_config);

        // Init registers that will change inside global loop
        let mut key_registers = SA::init_key(config.stage_config);
        let mut value_registers = SA::init_value(config.stage_config);
        let mut mask_registers = SA::init_mask(
            ComptimeOption::new_Some((seq_q, seq_kv)),
            config.stage_config,
        );
        let mut softmax_registers = SA::init_softmax(config.stage_config);
        let mut output_registers = SA::init_output(config.stage_config);

        // Init running state
        let mut stage_state = SA::init_state(config.stage_config);

        // Define number of global iterations
        let num_stage_iterations =
            seq_kv.div_ceil(config.stage_config.elements_in_partition_seq_kv());

        let mut barrier = ();

        // Global loop over seq_kv
        for _ in 0..num_stage_iterations {
            // Put key and value into stage
            key_reader.load_stage(&mut barrier, config.key_reader_config);
            value_reader.load_stage(&mut barrier, config.value_reader_config);

            sync_cube();

            // Core of flash attention
            SA::execute(
                &query_registers,
                &key_reader.stage(),
                &value_reader.stage(),
                &mut key_registers,
                &mut value_registers,
                &mask_reader,
                &mut mask_registers,
                &mut softmax_registers,
                &mut output_registers,
                &mut stage_state,
                config.stage_config,
            );

            sync_cube();

            // Advance in seq_kv direction
            key_reader.advance_view();
            value_reader.advance_view();
            mask_reader.advance_view();
        }

        // Accumulators must be rescaled using running state
        SA::rescale(&mut output_registers, stage_state, config.stage_config);

        // Write accumulators to output
        let mut out_stage = writer.stage();
        SA::write::<Self::Writer, Self::Config>(
            &output_registers,
            &mut out_stage,
            &mut writer,
            config.stage_config,
        )
    }

    fn init_query_reader(
        batch_index: u32,
        stage_q_offset: u32,
        query: VirtualTensor<QG<AP>, QGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> QueryReader<AP> {
        let layout = AttentionGlobalLayout::new(&query, batch_index, config.query_gmem_config);

        QueryReader::<AP>::new(stage_q_offset, query.view(layout), config.query_gmem_config)
    }

    fn init_key_reader(
        batch_index: u32,
        key: VirtualTensor<KG<AP>, KGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader {
        let step = config.stage_config.elements_in_partition_seq_kv().runtime();
        let layout =
            AttentionGlobalLayout::new(&key, batch_index, config.key_reader_config.gmem_config);
        FullStageGlobalReader::new(key.view(layout), (), step, config.key_reader_config)
    }

    fn init_value_reader(
        batch_index: u32,
        value: VirtualTensor<VG<AP>, VGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader {
        let step = config.stage_config.elements_in_partition_seq_kv().runtime();
        let layout =
            AttentionGlobalLayout::new(&value, batch_index, config.value_reader_config.gmem_config);
        FullStageGlobalReader::new(value.view(layout), (), step, config.value_reader_config)
    }

    fn init_mask_reader(
        batch_index: u32,
        stage_q_offset: u32,
        mask: ComptimeOption<VirtualTensor<MSK<AP>, MSKS<AP>>>,
        seq_kv_shape: u32,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader {
        let step = config.stage_config.elements_in_partition_seq_kv().runtime();
        let partition_q_offset = <SA::Partitioner as AttentionPartitioner>::seq_q_index()
            * config.stage_config.elements_in_partition_seq_q();

        #[comptime]
        match mask {
            ComptimeOption::Some(mask) => {
                let layout =
                    AttentionGlobalLayout::new(&mask, batch_index, config.mask_gmem_config);

                MaskReader::new_materialized(
                    stage_q_offset,
                    partition_q_offset,
                    mask.view(layout),
                    step,
                    seq_kv_shape,
                    config.mask_gmem_config,
                )
            }
            ComptimeOption::None => {
                MaskReader::new_logical(stage_q_offset + partition_q_offset, step)
            }
        }
    }

    fn init_writer(
        batch_index: u32,
        stage_q_offset: u32,
        out: VirtualTensor<OG<AP>, OGS<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let layout =
            AttentionGlobalLayout::new(&out, batch_index, config.writer_config.gmem_config);
        let out = out.view_mut(layout);

        Self::Writer::init::<SA::Config>(
            out.slice_mut((stage_q_offset, 0), out.shape()),
            config.writer_config,
        )
    }
}
