mod backward;
mod forward;

pub use cubek_tile::Residence;
pub use forward::InterpolateConfig;

pub(crate) use backward::interpolate_nearest_backward_launch;
pub(crate) use forward::interpolate_launch;
