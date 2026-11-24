use crate::kernels::tiled::roles::ViewOrder;
use crate::kernels::tiled::{ChannelAxis, ConvDecline, ConvGrouping, ConvRoles, SpatialDimCount};
use cubecl::prelude::TensorBinding;
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::MatmulOperand;

/// Refuses a tiled `tensor`: a rewrite of its axes by shape and strides alone reads a plain layout.
fn refuse_tiled(tensor: &TensorBinding) -> Result<(), Decline> {
    match tensor.tiling.is_tiled() {
        true => Err(ConvDecline::PermutedTiled.into()),
        false => Ok(()),
    }
}

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
    refuse_tiled(&tensor)?;
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
fn permute(
    order: ViewOrder,
    mut tensor: TensorBinding,
    spatial_dims: SpatialDimCount,
    grouping: ConvGrouping,
) -> Result<TensorBinding, Decline> {
    let buffer_dims = order.buffer_dims(spatial_dims, grouping);
    if buffer_dims.iter().copied().eq(0..tensor.shape.len()) {
        return Ok(tensor);
    }
    refuse_tiled(&tensor)?;
    tensor.shape = Shape::from(order.view(spatial_dims, grouping, &tensor.shape));
    tensor.strides = Strides::new(&order.view(spatial_dims, grouping, &tensor.strides));
    Ok(tensor)
}

/// `tensor`, the `operand` of the operation in its own layout, with its channels split by group
/// where the roles are grouped.
///
/// # Errors
///
/// [`ConvDecline::PermutedTiled`], where the channels split and `tensor` is tiled.
pub(super) fn split(
    roles: &ConvRoles,
    operand: MatmulOperand,
    tensor: TensorBinding,
) -> Result<TensorBinding, Decline> {
    split_channels(roles.channels.axis(operand), tensor, roles.grouping)
}

/// `rhs` with its dims listed in the order of [`ConvRoles::rhs`].
///
/// # Errors
///
/// [`ConvDecline::PermutedTiled`], where the dims move and `rhs` is tiled.
pub(super) fn rhs_view(roles: &ConvRoles, rhs: TensorBinding) -> Result<TensorBinding, Decline> {
    permute(roles.rhs_order, rhs, roles.spatial_dims, roles.grouping)
}

/// `out`, a tensor of the operation's output in its own layout, with its dims listed in the order
/// of [`ConvRoles::free`].
///
/// # Errors
///
/// [`ConvDecline::PermutedTiled`], where the dims move and `out` is tiled.
pub(super) fn out_view(roles: &ConvRoles, out: TensorBinding) -> Result<TensorBinding, Decline> {
    permute(roles.out_order, out, roles.spatial_dims, roles.grouping)
}

/// Whether `rhs` and `out`, viewed through `roles`, and the input's `extent` of `dimension`
/// (`None` where the input does not carry it) all give `dimension` the same extent.
pub(super) fn agree_on(
    roles: &ConvRoles,
    dimension: Dimension,
    extent: Option<usize>,
    [rhs, out]: [&TensorBinding; 2],
) -> bool {
    let (rhs_axes, out_axes) = (roles.rhs_axes(), roles.out_axes());
    let named = [(&rhs_axes[..], rhs), (&out_axes[..], out)];
    let mut extents = extent
        .into_iter()
        .chain(named.iter().filter_map(|(dimensions, tensor)| {
            let dim = dimensions.iter().position(|&named| named == dimension)?;
            Some(tensor.shape[dim])
        }));
    let first = extents.next();
    extents.all(|extent| Some(extent) == first)
}
