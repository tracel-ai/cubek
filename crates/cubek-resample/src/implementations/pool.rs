use crate::{
    components::{
        AccessPattern, Layout, LayoutExpand, NdLayout, NdLayoutLaunch, TapResult, dispatch_reader,
    },
    definition::{GlobalOperationKind, MemoryReaderKind},
    launch::ResampleArgs,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[derive(CubeType, Clone)]
pub struct PoolPattern {
    pub window_size: u32,
}

#[cube]
impl AccessPattern for PoolPattern {
    fn footprint_size(access_pattern: &Self) -> u32 {
        access_pattern.window_size
    }

    fn read_values<C: Numeric>(
        input: &Tensor<C>,
        in_layout: &NdLayout,
        out_coord: &CoordsDyn,
        tap_idx: u32,
        _access_pattern: &Self,
    ) -> TapResult<C> {
        // Naive 1D pooling offset for demonstration
        let mut in_coord = CoordsDyn::new();
        let rank = in_layout.strides.len();
        #[unroll]
        for i in 0..rank {
            let mut c = out_coord[i] * 2; // stride 2 assumption
            if i == rank - 1 {
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
pub fn pool_kernel<C: Numeric>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    window_size: u32,
    #[comptime] memory_reader_kind: MemoryReaderKind,
    #[comptime] operation_kind: GlobalOperationKind,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }
    let out_coord = out_layout.from_linear(index);
    let access_pattern = PoolPattern { window_size };

    dispatch_reader::<C, PoolPattern>(
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

pub struct PoolArgs {
    pub window_size: u32,
}

impl<R: Runtime> ResampleArgs<R> for PoolArgs {
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
            pool_kernel::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(32, 8),
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                out_layout,
                in_layout,
                self.window_size,
                memory_reader_kind,
                operation_kind,
                dtype,
            )
        };
    }
}
