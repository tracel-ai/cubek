use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
};
use cubek_std::tile::Tile;
use std::marker::PhantomData;

use crate::components::{
    global::simple::{MaskReader, QueryReader},
    stage::{MaskPartition, OutputPartition, partitioner::AttentionPartitioner},
};
use crate::{
    components::stage::{QueryPartition, SoftmaxPartition},
    components::tile::matmul::InnerMatmul,
    components::tile::output::AttentionOutput,
};
use crate::{
    components::{global::GlobalAttentionConfig, stage::PartitionAttentionConfig},
    definition::attention_types::*,
};
use crate::{
    components::{stage::KeyPartition, tile::TileAttention},
    components::{stage::ValuePartition, tile::softmax::Softmax},
    components::{stage::base::StageAttentionConfig, tile::TileAttentionConfig as _},
    {components::stage::StageAttention, definition::AttentionPrecision},
};
use cubecl::std::tensor::layout::Coords2d;

#[derive(CubeType)]
pub struct PartitionAttention<
    AP: AttentionPrecision,
    SK,
    SV,
    SO,
    TA: TileAttention<AP>,
    P: AttentionPartitioner,
> {
    #[cube(comptime)]
    _phantom: PhantomData<(AP, SK, SV, SO, TA, P)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    SK: Stage<KS<AP>, KSS<AP>, ReadOnly>,
    SV: Stage<VS<AP>, VSS<AP>, ReadOnly>,
    SO: Stage<OS<AP>, OSS<AP>, ReadWrite>,
    TA: TileAttention<AP>,
    P: AttentionPartitioner,
> StageAttention<AP> for PartitionAttention<AP, SK, SV, SO, TA, P>
{
    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PartitionAttentionConfig<TA::Config>;
    type Partitioner = P;

    type QueryPartition = QueryPartition<QT<AP>, QGS<AP>>;
    type KeyPartition = KeyPartition<KVT<AP>, KSS<AP>>;
    type ValuePartition = ValuePartition<KVT<AP>, VSS<AP>>;
    type SoftmaxPartition = SoftmaxPartition<SM<AP>, TA::Softmax>;
    type OutputPartition = OutputPartition<ACC<AP>, OSS<AP>, TA::Output>;
    type MaskPartition = MaskPartition<SM<AP>, TA::Softmax>;
    type RunningState = <TA::Softmax as Softmax<SM<AP>>>::RunningState;

    /// Executes the attention computation over one query–key/value partition.
    ///
    /// For each (q, kv) tile pair:
    /// 1. Computes attention scores across the full head dimension for that query row.
    /// 2. Applies masking and softmax locally to obtain unnormalized probabilities.
    /// 3. Uses these probabilities to partially accumulate the corresponding value tiles
    ///    into the output accumulators.
    fn execute(
        query_partition: &Self::QueryPartition,
        key_stage: &SK,
        value_stage: &SV,
        key_partition: &mut KeyPartition<KVT<AP>, KSS<AP>>,
        value_partition: &mut ValuePartition<KVT<AP>, VSS<AP>>,
        mask_reader: &MaskReader<AP>,
        mask_partition: &mut MaskPartition<SM<AP>, TA::Softmax>,
        softmax_partition: &mut SoftmaxPartition<SM<AP>, TA::Softmax>,
        output_partition: &mut OutputPartition<ACC<AP>, OSS<AP>, TA::Output>,
        state: &mut Sequence<Self::RunningState>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        let softmax_config = config.tile_config().softmax_config();

        let head_dim_factor =
            SM::<AP>::new(1.0 / ((p.head_dim * config.tile_size().head_dim) as f32).sqrt());

        // The problem is independent on each (q, kv) tile pair
        #[unroll]
        for kv in 0..p.seq_kv {
            #[unroll]
            for q in 0..p.seq_q as usize {
                // Get the q-th softmax tile and zero it
                softmax_partition.zero_score_at(q);

                // Get the only mask tile and fill it with q,kv-th data
                let mask_tile = mask_partition.get_mut();
                let (new_origin, mask_data) =
                    mask_reader.read::<P, Self::Config>((q as u32, kv), config);
                mask_tile.update(new_origin, mask_data);

                #[unroll]
                // Iterate over head dim to perform score matmul
                // Contrary to loop for value matmul, all iterations are accumulated into the same tile
                for hd in 0..p.head_dim as usize {
                    // Get the q,hd-th query which is always in registers
                    let query_tile = query_partition.get(q, hd, p.head_dim as usize);

                    // Get the only key-value tile and fill it with hd,kv-th key data
                    let key_tile = key_partition.get_mut();
                    let key_data = SK::tile(key_stage, (kv, hd as u32).runtime());
                    TA::ScoreMatmul::load_rhs(&key_data, &mut key_tile.tile);

                    // Perform score matmul on query and key, and accumulate in softmax tile
                    TA::ScoreMatmul::execute(
                        &query_tile.tile,
                        &key_tile.tile,
                        softmax_partition.get_score_mut(q),
                        config.tile_size().to_score_matmul_tile_size(),
                    );
                }

                // At this point, the softmax tile is filled with score

                // Get the q-th running state, i.e. the one associated with rows from q
                let state_q = state.index_mut(q);

                let scale = softmax_partition.softmax_at(
                    state_q,
                    mask_partition.get(),
                    head_dim_factor,
                    q,
                    softmax_config,
                );

                // At this point, the softmax tile is filled with probabilities

                #[unroll]
                // Iterate over val dim to perform value matmul
                // Contrary to loop for score matmul, all iterations contribute to different accumulators
                // The same accumulators will be accumulated to at the next kv iteration
                for vd in 0..p.val_dim as usize {
                    // Get the only key-value tile and fill it with hd,kv-th key data
                    let value_data = SV::tile(value_stage, (kv, vd as u32).runtime());
                    let value_tile = value_partition.get_mut();
                    TA::ValueMatmul::load_rhs(&value_data, &mut value_tile.tile);

                    // Scale the q,vd-th accumulator and scale it with previously obtained scale
                    let partition_val_dim = p.val_dim as usize;
                    output_partition.scale_mul_at(
                        &scale,
                        q,
                        vd,
                        partition_val_dim,
                        config.tile_config().output_config(),
                    );

                    // Perform value matmul on probabilities and values, and accumulate in accumulators
                    TA::ValueMatmul::execute(
                        softmax_partition.get_softmaxed_mut(q),
                        &value_partition.get().tile,
                        output_partition.get_at_mut(q, vd, partition_val_dim),
                        config.tile_size().to_value_matmul_tile_size(),
                    );
                }
            }
        }
    }

    fn rescale(
        acc: &mut OutputPartition<ACC<AP>, OSS<AP>, TA::Output>,
        state: Sequence<Self::RunningState>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        #[unroll]
        for q in 0..p.seq_q as usize {
            let running_state = state.index(q);

            #[unroll]
            for vd in 0..p.val_dim as usize {
                acc.scale_div_at(
                    running_state,
                    q,
                    vd,
                    config.shared().partition_size.val_dim as usize,
                    config.tile_config().output_config(),
                );
            }
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> Sequence<Self::RunningState> {
        let partition_seq_q = config.shared().partition_size.seq_q;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..partition_seq_q {
            sequence.push(TA::Softmax::init_state(
                config.tile_config().softmax_config(),
            ));
        }

        sequence
    }

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &mut OutputPartition<ACC<AP>, OSS<AP>, TA::Output>,
        stage: &mut SO,
        writer: &mut W,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        W::on_event(writer, WriteEvent::new_Begin());

        #[unroll]
        for q in 0..p.seq_q as usize {
            #[unroll]
            for vd in 0..p.val_dim as usize {
                let tile_pos = (q as u32 + P::seq_q_index() * p.seq_q, vd.runtime() as u32);
                let mut tile = SO::tile(stage, tile_pos);

                TA::Output::write_results(
                    acc.get_at_mut(q, vd, config.shared().partition_size.val_dim as usize),
                    &mut tile,
                    config.tile_config().output_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));
            }
        }

        W::on_event(writer, WriteEvent::new_Finish());
    }

    fn init_query(#[comptime] config: Self::Config) -> QueryPartition<QT<AP>, QGS<AP>> {
        QueryPartition::<QT<AP>, QGS<AP>>::new::<KVT<AP>, KSS<AP>, SM<AP>, Const<0>, TA::ScoreMatmul>(
            config.shared().partition_size,
            config.tile_config().score_matmul_config(),
        )
    }

    fn init_key(#[comptime] config: Self::Config) -> KeyPartition<KVT<AP>, KSS<AP>> {
        KeyPartition::<KVT<AP>, KSS<AP>>::new::<QT<AP>, QGS<AP>, SM<AP>, Const<0>, TA::ScoreMatmul>(
            config.tile_config().score_matmul_config(),
        )
    }

    fn init_value(#[comptime] config: Self::Config) -> ValuePartition<KVT<AP>, VSS<AP>> {
        ValuePartition::<KVT<AP>, VSS<AP>>::new::<
            SML<AP>,
            Const<0>,
            ACC<AP>,
            OSS<AP>,
            TA::ValueMatmul,
        >(config.tile_config().value_matmul_config())
    }

    fn init_softmax(#[comptime] config: Self::Config) -> SoftmaxPartition<SM<AP>, TA::Softmax> {
        SoftmaxPartition::<SM<AP>, TA::Softmax>::new(
            config.shared().partition_size,
            config.tile_config().softmax_config(),
        )
    }

    fn init_output(
        #[comptime] config: Self::Config,
    ) -> OutputPartition<ACC<AP>, OSS<AP>, TA::Output> {
        OutputPartition::<ACC<AP>, OSS<AP>, TA::Output>::new(
            config.shared().partition_size,
            config.tile_config().output_config(),
        )
    }

    fn init_mask(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] config: Self::Config,
    ) -> MaskPartition<SM<AP>, TA::Softmax> {
        MaskPartition::<SM<AP>, TA::Softmax>::new(
            out_of_bounds,
            config.tile_config().softmax_config(),
        )
    }

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut QueryPartition<QT<AP>, QGS<AP>>,
        #[comptime] config: Self::Config,
    ) {
        let partition_seq_q = config.shared().partition_size.seq_q;
        let partition_head_dim = config.shared().partition_size.head_dim;
        let attention_tile_size = config.tile_size();

        #[unroll]
        for q in 0..partition_seq_q as usize {
            #[unroll]
            for hd in 0..partition_head_dim as usize {
                let tile_to_write = registers.get_mut(q, hd, partition_head_dim as usize);
                let tile_read = reader.get_tile::<P>(
                    (q as u32, hd as u32).runtime(),
                    attention_tile_size,
                    partition_seq_q,
                    partition_head_dim,
                );

                TA::ScoreMatmul::load_lhs(
                    &Tile::new_SharedMemory(tile_read),
                    &mut tile_to_write.tile,
                );
            }
        }
    }
}
