use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::accelerated_whitebox::manual_matrix::{
    IdentA, IdentCD, ManualMatrix, MmaTypes,
};
use crate::components::tile::accelerated_whitebox::{ScoreMma, ValueMma};
use crate::definition::{AttentionPrecision, AttentionTileSize};

/// Trait for converting accumulator fragments into LHS fragments,
/// possibly using shared memory.
#[cube]
pub trait FragmentConvert<AP: AttentionPrecision>: CubeType {
    type Transit: CubeType + Copy;

    /// Convert accumulator into LHS fragment
    fn acc_to_lhs(
        acc: &ManualMatrix<IdentCD, ScoreMma<AP>>,
        lhs: &mut ManualMatrix<IdentA, ValueMma<AP>>,
        transit: &mut Self::Transit,
    );

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit;
}

#[derive(CubeType)]
pub struct RegisterFragmentConverter<AP: AttentionPrecision> {
    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}

#[cube]
impl<AP: AttentionPrecision> RegisterFragmentConverter<AP> {
    pub fn new(#[comptime] _tile_size: AttentionTileSize) -> Self {
        RegisterFragmentConverter::<AP> {
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision> FragmentConvert<AP> for RegisterFragmentConverter<AP> {
    type Transit = ();

    fn acc_to_lhs(
        acc: &ManualMatrix<IdentCD, ScoreMma<AP>>,
        lhs: &mut ManualMatrix<IdentA, ValueMma<AP>>,
        _transit: &mut Self::Transit,
    ) {
        assert!(
            acc.layout.num_rows == lhs.layout.num_rows
                && acc.layout.num_cols == lhs.layout.num_cols
        );

        #[unroll]
        for acc_row in 0..acc.layout.num_rows {
            #[unroll]
            for acc_col in 0..acc.layout.num_cols {
                let nth_acc = acc.layout.local_pos_to_nth((acc_row, acc_col).runtime());
                let val = acc.get_nth(nth_acc);
                let nth_lhs = lhs.layout.local_pos_to_nth((acc_row, acc_col).runtime());
                lhs.set_nth::<<ScoreMma<AP> as MmaTypes>::CD>(nth_lhs, val);
            }
        }
    }

    fn transit(
        #[comptime] _tile_size: AttentionTileSize,
        #[comptime] _num_planes: usize,
    ) -> Self::Transit {
        // Nothing to do
    }
}

#[derive(CubeType)]
pub struct SmemFragmentConverter<AP: AttentionPrecision> {
    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}

#[cube]
impl<AP: AttentionPrecision> SmemFragmentConverter<AP> {
    pub fn new(#[comptime] _tile_size: AttentionTileSize) -> Self {
        SmemFragmentConverter::<AP> {
            _phantom: PhantomData,
        }
    }
}
#[cube]
impl<AP: AttentionPrecision> FragmentConvert<AP> for SmemFragmentConverter<AP> {
    type Transit = SmemConvertTransit<AP::SoftmaxLhs>;

    fn acc_to_lhs(
        _acc: &ManualMatrix<IdentCD, ScoreMma<AP>>,
        _lhs: &mut ManualMatrix<IdentA, ValueMma<AP>>,
        _transit: &mut Self::Transit,
    ) {
        todo!()
        // let cast_fragment = cmma::cast::<Acc, Lhs>(&acc);
        // cmma::store(
        //     &mut transit.smem_slice,
        //     &cast_fragment,
        //     transit.stride,
        //     cmma::MatrixLayout::RowMajor,
        // );

        // sync_plane();

        // cmma::load(lhs, &transit.smem_slice.to_slice(), transit.stride)
    }

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit {
        let mut smem =
            SharedMemory::new((tile_size.seq_q * tile_size.seq_kv) as usize * num_planes);
        let smem_slot_size = tile_size.seq_q * tile_size.seq_kv;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let smem_slice = smem.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        SmemConvertTransit::<AP::SoftmaxLhs> {
            smem_slice,
            stride: tile_size.seq_kv,
        }
    }
}

#[derive(CubeType, Copy, Clone)]
#[allow(unused)]
pub struct SmemConvertTransit<E: Float> {
    smem_slice: SliceMut<E>,
    #[cube(comptime)]
    stride: u32,
}
