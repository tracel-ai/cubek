use cubecl::{
    std::tensor::{launch::BufferArg, layout::*},
    {prelude::*, std::tensor::launch::ViewLayoutLaunchArg},
};
use cubek_matmul::launch::BatchedCoords;

#[derive(CubeType)]
pub struct BiasLayout {
    shape: u32,
    #[cube(comptime)]
    vector_size: u32,
}

#[cube]
impl Layout for BiasLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, _, n) = pos;
        (n / self.vector_size) as usize
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, _, n) = pos;
        n < self.shape
    }

    fn shape(&self) -> Self::Coordinates {
        (1, 1, self.shape)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

impl ViewLayoutLaunchArg for BiasLayout {
    type RuntimeArg<R: Runtime> = ();
    type CompilationArg = ();

    fn register<R: Runtime, B: BufferArg>(
        _: Self::RuntimeArg<R>,
        buffer: &B,
        _: Type,
        launcher: &mut KernelLauncher<R>,
    ) {
        let shape = buffer.len();
        <u32 as LaunchArg>::register(shape as u32, launcher);
    }

    fn expand(
        _: &Self::CompilationArg,
        ty: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        BiasLayoutExpand {
            shape: <u32 as LaunchArg>::expand(&(), builder),
            vector_size: ty.vector_size() as u32,
        }
    }
}
