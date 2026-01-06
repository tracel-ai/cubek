//! User defines a [CubeCountStrategy], which, once problem is known,
//! becomes a [CubeCountPlan] where all information is known.
//! Then the [CubeCountPlan] is split into:
//! - The CubeCount
//! - The [CubeMapping] which maps a Cube to where it will work

mod mapping;
mod plan;
mod strategy;

pub use mapping::{CubeMapping, CubeMappingLaunch};
pub use plan::CubeCountPlan;
pub use strategy::CubeCountStrategy;
