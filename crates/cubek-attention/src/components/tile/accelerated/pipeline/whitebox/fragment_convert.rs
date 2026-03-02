use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::definition::AttentionTileSize;

/// Trait for converting accumulator fragments into LHS fragments,
/// possibly using shared memory.
#[cube]
pub trait FragmentConvert: CubeType {
    type Acc: Float;
    type Lhs: Float;
    type Transit: CubeType;

    /// Convert accumulator into LHS fragment
    fn acc_to_lhs(
        acc: &cmma::Matrix<Self::Acc>,
        lhs: &mut cmma::Matrix<Self::Lhs>,
        transit: &mut Self::Transit,
    );

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit;
}

#[derive(CubeType)]
pub struct RegisterFragmentConverter<Acc: Float, Lhs: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(Acc, Lhs)>,
}

#[cube]
impl<Acc: Float, Lhs: Float> RegisterFragmentConverter<Acc, Lhs> {
    pub fn new(#[comptime] _tile_size: AttentionTileSize) -> Self {
        RegisterFragmentConverter::<Acc, Lhs> {
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> FragmentConvert for RegisterFragmentConverter<Acc, Lhs> {
    type Acc = Acc;
    type Lhs = Lhs;
    type Transit = ();

    fn acc_to_lhs(
        acc: &cmma::Matrix<Self::Acc>,
        lhs: &mut cmma::Matrix<Self::Lhs>,
        _transit: &mut Self::Transit,
    ) {
        // TODO: implement register copy + cast
        todo!()
    }

    fn transit(
        #[comptime] _tile_size: AttentionTileSize,
        #[comptime] _num_planes: usize,
    ) -> Self::Transit {
        // Nothing to do
    }
}

#[derive(CubeType)]
pub struct SmemFragmentConverter<Acc: Float, Lhs: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(Acc, Lhs)>,
}

#[cube]
impl<Acc: Float, Lhs: Float> SmemFragmentConverter<Acc, Lhs> {
    pub fn new(#[comptime] _tile_size: AttentionTileSize) -> Self {
        SmemFragmentConverter::<Acc, Lhs> {
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> FragmentConvert for SmemFragmentConverter<Acc, Lhs> {
    type Acc = Acc;
    type Lhs = Lhs;
    type Transit = SmemConvertTransit<Lhs>;

    fn acc_to_lhs(
        acc: &cmma::Matrix<Self::Acc>,
        lhs: &mut cmma::Matrix<Self::Lhs>,
        transit: &mut Self::Transit,
    ) {
        let cast_fragment = cmma::cast::<Acc, Lhs>(&acc);
        cmma::store(
            &mut transit.smem_slice,
            &cast_fragment,
            transit.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_plane();

        cmma::load(lhs, &transit.smem_slice.to_slice(), transit.stride)
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

        SmemConvertTransit::<Lhs> {
            smem_slice,
            stride: tile_size.seq_kv,
        }
    }
}

#[derive(CubeType)]
pub struct SmemConvertTransit<E: Float> {
    smem_slice: SliceMut<E>,
    #[cube(comptime)]
    stride: u32,
}
