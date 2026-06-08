use crate::{
    components::{
        AccessPattern, Layout, LayoutExpand, NdLayout, NdLayoutLaunch, TapResult, dispatch_reader,
    },
    definition::{GlobalOperationKind, MemoryReaderKind},
    launch::ResampleArgs,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[derive(CubeType, Clone)]
pub struct InterpolatePattern {
    pub scale_factor: u32,
}

#[cube]
impl AccessPattern for InterpolatePattern {
    fn footprint_size(_access_pattern: &Self) -> u32 {
        1 // Nearest neighbor uses 1 tap
    }

    fn read_values<C: Numeric>(
        input: &Tensor<C>,
        in_layout: &NdLayout,
        out_coord: &CoordsDyn,
        _tap_idx: u32,
        access_pattern: &Self,
    ) -> TapResult<C> {
        let mut in_coord = CoordsDyn::new();
        let rank = in_layout.strides.len();
        // Nearest neighbor scaling
        #[unroll]
        for i in 0..rank {
            in_coord.push(out_coord[i] / access_pattern.scale_factor);
        }
        let input_idx = in_layout.to_source_pos(&in_coord);
        let value = input[input_idx];
        let weight = C::from_int(1);

        TapResult::<C> { value, weight }
    }
}

#[cube(launch_unchecked)]
pub fn interpolate_kernel<C: Numeric>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    #[comptime] memory_reader_kind: MemoryReaderKind,
    #[comptime] operation_kind: GlobalOperationKind,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }
    let out_coord = out_layout.from_linear(index);
    let access_pattern = InterpolatePattern { scale_factor: 2 };

    dispatch_reader::<C, InterpolatePattern>(
        input,
        output,
        in_layout,
        index,
        out_coord,
        access_pattern,
        memory_reader_kind,
        operation_kind,
    );
}

pub struct InterpolateArgs;

impl<R: Runtime> ResampleArgs<R> for InterpolateArgs {
    fn launch(
        self,
        client: &ComputeClient<R>,
        input: TensorBinding<R>,
        output: TensorBinding<R>,
        out_layout: NdLayoutLaunch<R>,
        in_layout: NdLayoutLaunch<R>,
        memory_reader_kind: MemoryReaderKind,
        operation_kind: GlobalOperationKind,
        dtype: StorageType,
    ) {
        unsafe {
            interpolate_kernel::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(32, 8),
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                out_layout,
                in_layout,
                memory_reader_kind,
                operation_kind,
                dtype,
            )
        };
    }
}
