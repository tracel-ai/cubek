use cubecl::prelude::*;

use crate::{InvalidConfigError, PlaneFlowConfig};

#[derive(Debug)]
/// Number of compute primitives required by some component, specified as either units, planes,
/// or a specialized plane-flow split.
pub enum CubeDimResource {
    Units(u32),
    Planes(u32),
    Specialized(PlaneFlowConfig),
}

impl CubeDimResource {
    /// Ensures [CubeDimResource] is the Planes variant, converting
    /// units using plane_dim, the number of units in a plane.
    ///
    /// Will fail if the number of units does not correspond to an exact number of planes
    pub fn as_plane_resource(self, plane_dim: u32) -> Result<Self, InvalidConfigError> {
        match self {
            CubeDimResource::Units(units) => {
                if units % plane_dim == 0 {
                    Ok(CubeDimResource::Planes(units / plane_dim))
                } else {
                    Err(Box::new(format!(
                        "Number of units {units:?} should be divisible by plane_dim {plane_dim:?}"
                    )))
                }
            }
            CubeDimResource::Planes(_) => Ok(self),
            CubeDimResource::Specialized(spec) => {
                Ok(CubeDimResource::Planes(spec.counts.total_count()))
            }
        }
    }

    /// Make a [CubeDim] from specified resources.
    ///
    /// Obtained CubeDim is always (plane_dim, number_of_planes, 1)
    ///
    /// Will fail if the number of units does not correspond to an exact number of planes
    pub fn to_cube_dim(self, plane_dim: u32) -> Result<CubeDim, InvalidConfigError> {
        match self {
            CubeDimResource::Units(_) => self.as_plane_resource(plane_dim)?.to_cube_dim(plane_dim),
            CubeDimResource::Planes(num_planes) => Ok(CubeDim::new_2d(plane_dim, num_planes)),
            CubeDimResource::Specialized(_) => {
                self.as_plane_resource(plane_dim)?.to_cube_dim(plane_dim)
            }
        }
    }

    /// Get the number of planes
    ///
    /// Will fail if the number of units does not correspond to an exact number of planes
    pub fn num_planes(self, plane_dim: u32) -> Result<u32, InvalidConfigError> {
        let plane_resources = self.as_plane_resource(plane_dim)?;
        if let CubeDimResource::Planes(num_planes) = plane_resources {
            Ok(num_planes)
        } else {
            unreachable!()
        }
    }

    /// Recover the [PlaneFlowConfig] view of this resource. `Units`/`Planes` produce a
    /// non-specialized config (all planes in the main flow).
    pub fn as_specialized(self, plane_dim: u32) -> Result<PlaneFlowConfig, InvalidConfigError> {
        match self {
            CubeDimResource::Units(_) => {
                self.as_plane_resource(plane_dim)?.as_specialized(plane_dim)
            }
            CubeDimResource::Planes(num_planes) => {
                Ok(PlaneFlowConfig::new_unspecialized(num_planes))
            }
            CubeDimResource::Specialized(spec) => Ok(spec),
        }
    }
}
