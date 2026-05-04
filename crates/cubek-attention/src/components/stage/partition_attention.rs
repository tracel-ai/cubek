use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
};
use cubek_std::tile::{RowWise, SharedTile, Tile};
use std::marker::PhantomData;

use crate::components::stage::partition::init_running_state;
use crate::components::stage::{QueryPartition, SoftmaxPartition};
use crate::components::tile::MaskConfig;
use crate::components::{
    global::simple::{MaskReader, QueryReader},
    stage::{MaskPartition, OutputPartition, partitioner::AttentionPartitioner},
};
use crate::{
    components::stage::KeyPartition,
    components::stage::ValuePartition,
    components::stage::base::StageAttentionConfig,
    {components::stage::StageAttention, definition::AttentionPrecision},
};
use crate::{
    components::{global::GlobalAttentionConfig, stage::PartitionAttentionConfig},
    definition::attention_types::*,
};
use cubecl::std::tensor::layout::Coords2d;

#[derive(CubeType)]
pub struct PartitionAttention<AP: AttentionPrecision, SK, SV, SO, P: AttentionPartitioner> {
    #[cube(comptime)]
    _phantom: PhantomData<(AP, SK, SV, SO, P)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    SK: Stage<KS<AP>, ReadOnly>,
    SV: Stage<VS<AP>, ReadOnly>,
    SO: Stage<OS<AP>, ReadWrite>,
    P: AttentionPartitioner,
> StageAttention<AP> for PartitionAttention<AP, SK, SV, SO, P>
{
    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PartitionAttentionConfig;
    type Partitioner = P;

    type QueryPartition = QueryPartition<QT<AP>>;
    type KeyPartition = KeyPartition<KVT<AP>>;
    type ValuePartition = ValuePartition<KVT<AP>>;
    type SoftmaxPartition = SoftmaxPartition<SM<AP>, SML<AP>>;
    type OutputPartition = OutputPartition<ACC<AP>>;
    type MaskPartition = MaskPartition<SM<AP>>;
    type RunningState = (RowWise<SM<AP>>, RowWise<SM<AP>>);

    fn execute(
        query_partition: &Self::QueryPartition,
        key_stage: &SK,
        value_stage: &SV,
        key_partition: &mut KeyPartition<KVT<AP>>,
        value_partition: &mut ValuePartition<KVT<AP>>,
        mask_reader: &MaskReader<AP>,
        mask_partition: &mut MaskPartition<SM<AP>>,
        softmax_partition: &mut SoftmaxPartition<SM<AP>, SML<AP>>,
        output_partition: &mut OutputPartition<ACC<AP>>,
        state: &mut Sequence<Self::RunningState>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        let head_dim_factor =
            SM::<AP>::new(1.0 / ((p.head_dim * config.tile_size().head_dim) as f32).sqrt());

        #[unroll]
        for kv in 0..p.seq_kv {
            #[unroll]
            for q in 0..p.seq_q as usize {
                softmax_partition.zero_score_at(q);

                let mask_tile = mask_partition.get_mut();
                let (new_origin, mask_data) =
                    mask_reader.read::<P, Self::Config>((q as u32, kv), config);
                mask_tile.update(new_origin, mask_data);

                #[unroll]
                for hd in 0..p.head_dim as usize {
                    let query_tile = query_partition.get(q, hd, p.head_dim as usize);

                    let key_tile = key_partition.get_mut();
                    let key_data = SK::tile(key_stage, (kv, hd as u32).runtime());

                    key_tile
                        .tile
                        .copy_from::<KS<AP>, KSS<AP>, QT<AP>, KVT<AP>, SM<AP>, ReadOnly>(
                            &key_data,
                            cubek_std::StageIdent::Rhs,
                        );

                    softmax_partition
                        .get_score_mut(q)
                        .mma(&query_tile.tile, &key_tile.tile);
                }

                let state_q = state.index_mut(q);

                let scale =
                    softmax_partition.softmax_at(state_q, mask_partition.get(), head_dim_factor, q);

                #[unroll]
                for vd in 0..p.val_dim as usize {
                    let value_data = SV::tile(value_stage, (kv, vd as u32).runtime());
                    let value_tile = value_partition.get_mut();

                    value_tile
                        .tile
                        .copy_from::<VS<AP>, VSS<AP>, SML<AP>, KVT<AP>, ACC<AP>, ReadOnly>(
                            &value_data,
                            cubek_std::StageIdent::Rhs,
                        );

                    let partition_val_dim = p.val_dim as usize;
                    output_partition.scale_mul_at::<SM<AP>>(&scale, q, vd, partition_val_dim);

                    output_partition.get_at_mut(q, vd, partition_val_dim).mma(
                        softmax_partition.get_softmaxed_mut(q),
                        &value_partition.get().tile,
                    );
                }
            }
        }
    }

    fn rescale(
        acc: &mut OutputPartition<ACC<AP>>,
        state: Sequence<Self::RunningState>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        #[unroll]
        for q in 0..p.seq_q as usize {
            let running_state = state.index(q);

            #[unroll]
            for vd in 0..p.val_dim as usize {
                acc.scale_div_at::<SM<AP>>(
                    &running_state.1,
                    q,
                    vd,
                    config.shared().partition_size.val_dim as usize,
                );
            }
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> Sequence<Self::RunningState> {
        let partition_seq_q = config.shared().partition_size.seq_q;
        let mut sequence = Sequence::new();
        let softmax_kind = config.tile_attention().softmax_kind();

        #[unroll]
        for _ in 0..partition_seq_q {
            sequence.push(init_running_state::<SM<AP>>(softmax_kind));
        }

        sequence
    }

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &mut OutputPartition<ACC<AP>>,
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

                let acc_tile =
                    acc.get_at_mut(q, vd, config.shared().partition_size.val_dim as usize);
                acc_tile.write_results::<OS<AP>, OSS<AP>>(&mut tile);

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));
            }
        }

        W::on_event(writer, WriteEvent::new_Finish());
    }

    fn init_query(#[comptime] config: Self::Config) -> QueryPartition<QT<AP>> {
        QueryPartition::<QT<AP>>::new(
            config.shared().partition_size,
            config.tile_attention().score_matmul(),
        )
    }

    fn init_key(#[comptime] config: Self::Config) -> KeyPartition<KVT<AP>> {
        KeyPartition::<KVT<AP>>::new(config.tile_attention().score_matmul())
    }

    fn init_value(#[comptime] config: Self::Config) -> ValuePartition<KVT<AP>> {
        ValuePartition::<KVT<AP>>::new(config.tile_attention().value_matmul())
    }

    fn init_softmax(#[comptime] config: Self::Config) -> SoftmaxPartition<SM<AP>, SML<AP>> {
        SoftmaxPartition::<SM<AP>, SML<AP>>::new(
            config.shared().partition_size,
            config.tile_attention().score_matmul(),
            config.tile_attention().value_matmul(),
        )
    }

    fn init_output(#[comptime] config: Self::Config) -> OutputPartition<ACC<AP>> {
        OutputPartition::<ACC<AP>>::new(
            config.shared().partition_size,
            config.tile_attention().value_matmul(),
        )
    }

    fn init_mask(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] config: Self::Config,
    ) -> MaskPartition<SM<AP>> {
        let mask_config = comptime! {
            MaskConfig {
                layout: config.tile_attention().mask_layout(),
                causal: config.tile_attention().causal_mask(),
                materialized: config.tile_attention().materialized_mask(),
            }
        };
        MaskPartition::<SM<AP>>::new(out_of_bounds, mask_config)
    }

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut QueryPartition<QT<AP>>,
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

                tile_to_write
                    .tile
                    .copy_from::<QG<AP>, QGS<AP>, QT<AP>, KVT<AP>, SM<AP>, ReadOnly>(
                        &Tile::new_SharedMemory(SharedTile::wrap::<QGS<AP>>(tile_read)),
                        cubek_std::StageIdent::Lhs,
                    );
            }
        }
    }
}
