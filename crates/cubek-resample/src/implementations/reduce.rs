use crate::{
    components::{
        AccessPattern, Layout, LayoutExpand, NdLayout, NdLayoutLaunch, TapResult, dispatch_reader,
    },
    definition::{GlobalOperationKind, MemoryReaderKind},
    launch::ResampleArgs,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[derive(CubeType, Clone)]
pub struct ReduceAxisPattern {
    pub reduce_size: u32,
    pub reduce_axis: u32,
}

#[cube]
impl AccessPattern for ReduceAxisPattern {
    fn footprint_size(access_pattern: &Self) -> u32 {
        access_pattern.reduce_size
    }

    fn read_values<C: Numeric>(
        input: &Tensor<C>,
        in_layout: &NdLayout,
        out_coord: &CoordsDyn,
        tap_idx: u32,
        access_pattern: &Self,
    ) -> TapResult<C> {
        // Read out_coord, then offset by tap_idx along reduce_axis
        let mut in_coord = CoordsDyn::new();
        let rank = in_layout.strides.len();
        #[unroll]
        for i in 0..rank {
            let mut c = out_coord[i];
            if i == access_pattern.reduce_axis as usize {
                c += tap_idx;
            }
            in_coord.push(c);
        }
        let input_idx = in_layout.to_source_pos(&in_coord);
        let value = input[input_idx];
        let weight = C::from_int(1);

        TapResult::<C> { value, weight }
    }
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<C: Numeric>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    reduce_size: u32,
    reduce_axis: u32,
    #[comptime] memory_reader_kind: MemoryReaderKind,
    #[comptime] operation_kind: GlobalOperationKind,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }
    let out_coord = out_layout.from_linear(index);
    let access_pattern = ReduceAxisPattern {
        reduce_size,
        reduce_axis,
    };

    dispatch_reader::<C, ReduceAxisPattern>(
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

pub struct ReduceArgs {
    pub reduce_size: u32,
    pub reduce_axis: u32,
}

impl<R: Runtime> ResampleArgs<R> for ReduceArgs {
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
            reduce_kernel::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(32, 8),
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                out_layout,
                in_layout,
                self.reduce_size,
                self.reduce_axis,
                memory_reader_kind,
                operation_kind,
                dtype,
            )
        };
    }
}
