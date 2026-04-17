use crate::components::tile::Tilex;
use crate::definition::{MatmulTypes, MatrixTypes};
use crate::{
    components::{stage::Stage, tile::TileMatmul},
    definition::{Acc, StageSize},
};
use cubecl::prelude::*;
use cubek_std::{MatrixLayout, PartitionSize};

#[derive(CubeType)]
/// Wrapper over a sequence of Tile Matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<
    MP: MatmulTypes,
    TM: TileMatmul<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Lhs as MatrixTypes>::RegisterSize,
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::RegisterSize,
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
        >,
> {
    sequence: Sequence<
        Tilex<
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
            ReadWrite,
        >,
    >,
    #[cube(comptime)]
    _phantom: std::marker::PhantomData<TM>,
}

type StageTy<T> = crate::definition::Stage<T>;

#[cube]
impl<
    MP: MatmulTypes,
    TM: TileMatmul<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Lhs as MatrixTypes>::RegisterSize,
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::RegisterSize,
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
        >,
> Accumulators<MP, TM>
{
    /// Create a new accumulators sequence from the provided configuration
    pub fn new(
        #[comptime] partition_size: PartitionSize,
        #[comptime] acc_layout: MatrixLayout,
        #[comptime] tile_config: TM::Config,
    ) -> Accumulators<MP, TM> {
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.mn() {
            accumulators.push(TM::allocate_acc(acc_layout, tile_config));
        }

        Accumulators::<MP, TM> {
            sequence: accumulators,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Load all accumulators from the specified stage
    pub fn load<R: Stage<StageTy<Acc<MP>>, StageSize<Acc<MP>>, ReadOnly>>(
        &mut self,
        stage: &R,
        #[comptime] tiles_in_stage_partition_m: usize,
        #[comptime] tiles_in_stage_partition_n: usize,
        #[comptime] tile_config: TM::Config,
    ) {
        #[unroll]
        for m in 0..tiles_in_stage_partition_m {
            #[unroll]
            for n in 0..tiles_in_stage_partition_n {
                let acc = self.get_at_mut(m, n, tiles_in_stage_partition_n);
                let tile = R::tile(stage, (m as u32, n as u32).runtime());
                TM::load_acc(&tile, acc, tile_config);
            }
        }
    }

    /// Fetch a reference to the accumulator at (`m`, `n`)
    pub fn get_at(
        &self,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] tiles_in_stage_partition_n: usize,
    ) -> &Tilex<<MP::Acc as MatrixTypes>::Register, <MP::Acc as MatrixTypes>::RegisterSize, ReadWrite>
    {
        &self.sequence[m * tiles_in_stage_partition_n + n]
    }

    /// Fetch a mutable reference to the accumulator at (`m`, `n`)
    pub fn get_at_mut(
        &mut self,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] tiles_in_stage_partition_n: usize,
    ) -> &mut Tilex<
        <MP::Acc as MatrixTypes>::Register,
        <MP::Acc as MatrixTypes>::RegisterSize,
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
