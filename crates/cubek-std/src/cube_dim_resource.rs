use cubecl::prelude::*;

use crate::InvalidConfigError;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Represents how many planes are used for main computation and for loading-only tasks.
pub struct PlaneFlowCounts {
    /// Number of planes participating in main flow and (possibly) loading.
    pub main_flow: u32,
    /// Number of planes dedicated solely to loading.
    pub load_only: u32,
}

impl PlaneFlowCounts {
    /// Return the total number of planes
    pub fn total_count(&self) -> u32 {
        self.main_flow + self.load_only
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// How planes are partitioned by id between the main flow and load-only roles.
pub enum PlaneFlowPartitionRule {
    MainFlowOnly,
    LoadOnlyFirst { load_only: u32 },
    LoadOnlyLast { main_flow: u32 },
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Plane-flow configuration carried by [`CubeDimResource::Specialized`]. Holds the
/// counts for main-flow vs load-only planes and the partition rule used at runtime.
pub struct SpecializedCubeDim {
    pub counts: PlaneFlowCounts,
    pub partition_rule: PlaneFlowPartitionRule,
}

impl SpecializedCubeDim {
    /// All planes participate in the main flow; no load-only planes.
    pub fn new_unspecialized(num_planes: u32) -> Self {
        Self {
            counts: PlaneFlowCounts {
                main_flow: num_planes,
                load_only: 0,
            },
            partition_rule: PlaneFlowPartitionRule::MainFlowOnly,
        }
    }

    /// Number of planes participating in main flow.
    pub fn main_flow_count(&self) -> u32 {
        self.counts.main_flow
    }

    /// Whether the configuration uses dedicated load-only planes.
    pub fn has_specialization(&self) -> bool {
        self.counts.load_only > 0
    }
}

#[derive(Debug)]
/// Number of compute primitives required by some component, specified as either units, planes,
/// or a specialized plane-flow split.
pub enum CubeDimResource {
    Units(u32),
    Planes(u32),
    Specialized(SpecializedCubeDim),
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

    /// Recover the [SpecializedCubeDim] view of this resource. `Units`/`Planes` produce a
    /// non-specialized config (all planes in the main flow).
    pub fn as_specialized(self, plane_dim: u32) -> Result<SpecializedCubeDim, InvalidConfigError> {
        match self {
            CubeDimResource::Units(_) => {
                self.as_plane_resource(plane_dim)?.as_specialized(plane_dim)
            }
            CubeDimResource::Planes(num_planes) => {
                Ok(SpecializedCubeDim::new_unspecialized(num_planes))
            }
            CubeDimResource::Specialized(spec) => Ok(spec),
        }
    }
}
