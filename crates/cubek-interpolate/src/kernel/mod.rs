pub mod backward;
pub mod forward;

pub use forward::InterpolateConfig;

pub(crate) use backward::interpolate_nearest_backward_launch;
