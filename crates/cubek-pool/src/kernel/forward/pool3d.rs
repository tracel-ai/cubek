use cubecl::{
    prelude::{TensorBinding, *},
    std::tensor::{
        launch::ViewArg,
        layout::fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
    },
};

pub(crate) type Position3d = (usize, usize, usize, usize, usize);

pub(crate) fn view5d(tensor: TensorBinding, vector_size: VectorSize) -> ViewArg<Position3d> {
    let shape = (
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
        tensor.shape[4],
    );
    let layout = FixedDimLayoutLaunch::<Position3d>::from_shape_handle_unchecked(
        &tensor,
        shape,
        vector_size,
    );
    let buffer = tensor.into_tensor_arg();
    ViewArg::new_tensor::<FixedDimLayout<Position3d>>(buffer, layout)
}
