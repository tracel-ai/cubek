use std::marker::PhantomData;

use crate::components::stage::matmul::scheduler::PartitionScheduler;
use crate::components::tile_matmul::{
    Scope, Tile, TileMatmul, cmma_allocate_acc, interleaved_allocate_acc, mma_allocate_acc,
    planevec_allocate_acc, register_allocate_acc,
};
use crate::definition::{
    AccRE, AccRS, AccSE, AccSS, LhsRE, MatmulTypes, MatrixTypes, RhsRE, StageIdent,
};
use crate::{
    components::stage::Stage,
    definition::{Acc, StageSize},
};
use cubecl::prelude::*;
use cubek_std::{MatrixLayout, PartitionSize};

#[derive(CubeType)]
/// Wrapper over a sequence of Tile Matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<MP: MatmulTypes, Sc: Scope> {
    sequence: Sequence<
        Tile<
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
            Sc,
            ReadWrite,
        >,
    >,
    #[cube(comptime)]
    _phantom: PhantomData<Sc>,
}

type StageTy<T> = crate::definition::Stage<T>;

#[cube]
impl<MT: MatmulTypes, Sc: Scope> Accumulators<MT, Sc> {
    /// Create a new accumulators sequence from the provided configuration
    pub fn new(
        #[comptime] partition_size: PartitionSize,
        #[comptime] acc_layout: MatrixLayout,
        #[comptime] tile_config: TileMatmul,
    ) -> Accumulators<MT, Sc> {
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.mn() {
            accumulators.push(allocate_acc::<MT, Sc>(acc_layout, tile_config));
        }

        Accumulators::<MT, Sc> {
            sequence: accumulators,
            _phantom: PhantomData,
        }
    }

    /// Load all accumulators from the specified stage
    pub fn load<R: Stage<StageTy<Acc<MT>>, StageSize<Acc<MT>>, ReadOnly>>(
        &mut self,
        stage: &R,
        partition_scheduler: &PartitionScheduler,
        #[comptime] tiles_in_stage_partition_m: usize,
        #[comptime] tiles_in_stage_partition_n: usize,
    ) {
        #[unroll]
        for m in 0..tiles_in_stage_partition_m {
            let m_stage = partition_scheduler.map_m(m as u32);

            #[unroll]
            for n in 0..tiles_in_stage_partition_n {
                let n_stage = partition_scheduler.map_n(n as u32);

                let acc = self.get_at_mut(m, n, tiles_in_stage_partition_n);
                let tile = R::tile::<Sc>(stage, (m_stage, n_stage));
                acc.copy_from::<AccSE<MT>, AccSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
                    &tile,
                    StageIdent::Acc,
                );
            }
        }
    }

    /// Fetch a reference to the accumulator at (`m`, `n`)
    pub fn get_at(
        &self,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] tiles_in_stage_partition_n: usize,
    ) -> &Tile<
        <MT::Acc as MatrixTypes>::Register,
        <MT::Acc as MatrixTypes>::RegisterSize,
        Sc,
        ReadWrite,
    > {
        &self.sequence[m * tiles_in_stage_partition_n + n]
    }

    /// Fetch a mutable reference to the accumulator at (`m`, `n`)
    pub fn get_at_mut(
        &mut self,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] tiles_in_stage_partition_n: usize,
    ) -> &mut Tile<
        <MT::Acc as MatrixTypes>::Register,
        <MT::Acc as MatrixTypes>::RegisterSize,
        Sc,
        ReadWrite,
    > {
        self.sequence.index_mut(m * tiles_in_stage_partition_n + n)
    }
}

#[derive(CubeType)]
/// Rhs tiles, can be doubled for partition double buffering
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}

#[cube]
fn allocate_acc<MT: MatmulTypes, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: TileMatmul,
) -> Tile<AccRE<MT>, AccRS<MT>, Sc, ReadWrite> {
    match config {
        TileMatmul::Cmma(c) => cmma_allocate_acc::<AccRE<MT>, AccRS<MT>, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => {
            mma_allocate_acc::<AccRE<MT>, AccRS<MT>, LhsRE<MT>, RhsRE<MT>, Sc>(layout, c)
        }
        TileMatmul::Register(c) => register_allocate_acc::<AccRE<MT>, AccRS<MT>, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_acc::<AccRE<MT>, AccRS<MT>, Sc>(layout, c),
        TileMatmul::Interleaved(c) => {
            interleaved_allocate_acc::<AccRE<MT>, AccRS<MT>, Sc>(layout, c)
        }
    }
}
