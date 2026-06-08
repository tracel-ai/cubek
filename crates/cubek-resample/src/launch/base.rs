use crate::{
    components::NdLayoutLaunch,
    definition::{GlobalOperationKind, MemoryReaderKind},
};
use cubecl::prelude::*;

pub trait ResampleArgs<R: Runtime> {
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
    );
}
