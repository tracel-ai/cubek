use crate::kernels::tiled::roles::ViewOrder;
use crate::kernels::tiled::{ChannelAxis, ConvDecline, ConvGrouping, ConvRoles, SpatialDimCount};
use cubecl::prelude::TensorBinding;
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubecl::zspace::{Shape, Strides};

/// `tensor` with its channel axis cut into `[groups, channels / groups]` where `grouping` is
/// grouped, by shape and strides alone. An axis the groups do not divide is the caller's to
/// have refused.
///
/// # Errors
///
/// [`ConvDecline::PermutedTiled`], where `grouping` is grouped and `tensor` is tiled.
pub(in crate::kernels::tiled) fn split_channels(
    axis: ChannelAxis,
    mut tensor: TensorBinding,
    grouping: ConvGrouping,
) -> Result<TensorBinding, Decline> {
    let ConvGrouping::Grouped(groups) = grouping else {
        return Ok(tensor);
    };
    if tensor.tiling.is_tiled() {
        return Err(ConvDecline::PermutedTiled.into());
    }
    let axis = match axis {
        ChannelAxis::First => 0,
        ChannelAxis::Last => tensor.shape.len() - 1,
    };
    let (extent, stride) = (tensor.shape[axis], tensor.strides[axis]);
    let per_group = extent / groups;
    let (mut shape, mut strides) = (tensor.shape.to_vec(), tensor.strides.to_vec());
    shape.splice(axis..=axis, [groups, per_group]);
    strides.splice(axis..=axis, [stride * per_group, stride]);
    tensor.shape = Shape::from(shape);
    tensor.strides = Strides::new(&strides);
    Ok(tensor)
}

/// `tensor` with its dims listed in the order of `order`.
///
/// # Errors
///
/// [`ConvDecline::PermutedTiled`], where the dims move and `tensor` is tiled.
pub(in crate::kernels::tiled) fn permute(
    order: ViewOrder,
    mut tensor: TensorBinding,
    spatial_dims: SpatialDimCount,
    grouping: ConvGrouping,
) -> Result<TensorBinding, Decline> {
    let buffer_dims = order.buffer_dims(spatial_dims, grouping);
    if buffer_dims.iter().copied().eq(0..tensor.shape.len()) {
        return Ok(tensor);
    }
    if tensor.tiling.is_tiled() {
        return Err(ConvDecline::PermutedTiled.into());
    }
    tensor.shape = Shape::from(order.view(spatial_dims, grouping, &tensor.shape));
    tensor.strides = Strides::new(&order.view(spatial_dims, grouping, &tensor.strides));
    Ok(tensor)
}

/// Whether `rhs` and `out`, viewed through `roles`, and the input's `extent` of `dimension`
/// (`None` where the input does not carry it) all give `dimension` the same extent.
pub(super) fn agree_on(
    roles: &ConvRoles,
    dimension: Dimension,
    extent: Option<usize>,
    [rhs, out]: [&TensorBinding; 2],
) -> bool {
    let free = roles.free.in_order();
    let named = [(&roles.rhs[..], rhs), (&free[..], out)];
    let mut extents = extent
        .into_iter()
        .chain(named.iter().filter_map(|(dimensions, tensor)| {
            let dim = dimensions.iter().position(|&named| named == dimension)?;
            Some(tensor.shape[dim])
        }));
    let first = extents.next();
    extents.all(|extent| Some(extent) == first)
}
