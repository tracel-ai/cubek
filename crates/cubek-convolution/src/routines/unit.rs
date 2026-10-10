use cubecl::{client::Client, ir::ElemType, prelude::TensorBinding, server::LaunchError};
use cubek_matmul::definition::AvailableVectorSizes;
use cubek_matmul::multi_level::{
    args::TensorArgs,
    components::global::read::sync_full_cyclic::SyncFullCyclicLoading,
    definition::BatchMatmulBlueprint,
    routines::batch::simple_unit::{SimpleUnitAlgorithm, SimpleUnitSelectionArgs},
    tile::{ColMajorTilingOrder, RowMajorTilingOrder},
};

use crate::{
    components::{
        ConvolutionOperation,
        global::{args::RuntimeArgs, read::strategy::sync_bias::SyncBiasLoading},
    },
    routines::{Routine, contiguous_pitched_layout},
};

/// Register-tiled implicit convolution using ordinary SIMT arithmetic.
///
/// Reuses the convolution window layouts without requiring accelerated matrix
/// instructions or materializing an im2col matrix in device memory.
pub struct UnitConv;

pub(crate) type UnitMatmul = SimpleUnitAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncBiasLoading,
>;

impl Routine for UnitConv {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleUnitSelectionArgs;
    type MatmulRoutine = UnitMatmul;
    type Args = TensorArgs<RuntimeArgs>;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }

    fn filter_vector_sizes(mut sizes: AvailableVectorSizes) -> AvailableVectorSizes {
        // The fixed F32 tiles are four elements wide. Keep the same contract
        // on devices that expose wider I/O vectors.
        sizes.lhs.retain(|v| *v <= 4);
        sizes.rhs.retain(|v| *v <= 4);
        sizes.out.retain(|v| *v <= 4);
        sizes
    }
}
