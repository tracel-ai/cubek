//! One kernel launch: the [`Launcher`].

use cubecl::ir::OpaqueType;
use cubecl::prelude::*;

use crate::{
    Arg, Axis, Geometry, Partitioning, PartitioningLaunch, Refusal, Space, SpaceLaunch, Unlabelled,
};

/// How many cubes of how many units the launch runs: stated, or read off the levels.
#[derive(Clone, Debug)]
pub enum Grid {
    Stated {
        cube_count: CubeCount,
        cube_dim: CubeDim,
    },
    FromLevels,
}

/// One launch: a kernel-form partitioning, the concrete space with real extents, and the grid.
#[derive(Clone)]
pub struct Launcher {
    client: Client,
    partitioning: Partitioning,
    concrete: Space,
    cube_count: CubeCount,
    cube_dim: CubeDim,
}

impl Launcher {
    /// `partitioning` over `concrete`'s real extents, on `grid`, or the [`Refusal`] saying why the
    /// device cannot run it.
    pub fn new(
        client: &Client,
        partitioning: Partitioning,
        concrete: &Space,
        grid: Grid,
    ) -> Result<Self, Refusal> {
        let (cube_count, cube_dim) = match grid {
            Grid::Stated {
                cube_count,
                cube_dim,
            } => (cube_count, cube_dim),
            Grid::FromLevels => {
                let plane_size = client.properties().hardware.plane_size_max;
                let plane_units = partitioning.units();
                if plane_units != 1 && plane_units != plane_size {
                    return Err(Refusal::GridNotReadOffLevels {
                        units: plane_units,
                        plane: plane_size,
                    });
                }
                // The kernel-form space may state no extents.
                let over_concrete =
                    Partitioning::new(concrete.clone(), partitioning.levels().to_vec());
                (
                    over_concrete.cube_count(),
                    over_concrete.cube_dim(plane_size),
                )
            }
        };
        let max_units = client.properties().hardware.max_units_per_cube;
        if cube_dim.num_elems() > max_units {
            return Err(Refusal::CubePastDevice {
                units: cube_dim.num_elems(),
                most: max_units,
            });
        }
        let fillers = partitioning.fillers();
        // Checked on the host: a panic at expansion is unseen and the launch returns zeros.
        let barriers = client
            .properties()
            .features
            .types
            .opaque
            .contains(&OpaqueType::Barrier);
        if fillers > 0 && !barriers {
            return Err(Refusal::FillersWithoutBarriers { fillers });
        }
        Ok(Launcher {
            client: client.clone(),
            partitioning,
            concrete: concrete.clone(),
            cube_count,
            cube_dim,
        })
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn cube_count(&self) -> CubeCount {
        self.cube_count.clone()
    }

    pub fn cube_dim(&self) -> CubeDim {
        self.cube_dim
    }

    /// The concrete space: this launch's real extents.
    pub fn space(&self) -> &Space {
        &self.concrete
    }

    /// The partitioning this launch states, its space in kernel form.
    pub fn partitioning(&self) -> &Partitioning {
        &self.partitioning
    }

    /// The kernel's partitioning argument, dynamic extents sized by this launch.
    pub fn partitioning_arg(&self) -> PartitioningLaunch {
        PartitioningLaunch::new(self.space_arg(), self.partitioning.levels().to_vec())
    }

    /// The kernel-form space as a kernel argument.
    fn space_arg(&self) -> SpaceLaunch {
        let kernel = self.partitioning.space();
        let mut sizes = SequenceArg::new();
        if !kernel.is_static() {
            for axis in kernel.axes() {
                sizes.push(self.concrete.extent(axis));
            }
        }
        SpaceLaunch::new(kernel.shape().clone(), sizes)
    }

    /// The partitioning's levels over the concrete space.
    fn over_concrete(&self) -> Partitioning {
        Partitioning::new(self.concrete.clone(), self.partitioning.levels().to_vec())
    }

    /// The axes some tile reaches past the end of, whose accesses are masked.
    pub(crate) fn overhangs(&self) -> Vec<Axis> {
        self.over_concrete().overhanging()
    }

    /// The leaf edge along `axis`: the stated tile, or the whole extent where none was.
    fn leaf_edge(&self, axis: Axis) -> usize {
        self.over_concrete()
            .leaf()
            .extents()
            .iter()
            .find(|&&(a, _)| a == axis)
            .map(|&(_, edge)| edge)
            .unwrap_or_else(|| self.concrete.extent(axis))
    }

    /// Bind `binding` as an operand of this launch.
    pub fn arg(&self, binding: TensorBinding) -> Arg<'_, Unlabelled> {
        Arg::bound(self, binding)
    }

    /// [`arg`](Self::arg) over a stated geometry, for an operand with no tensor; end it with
    /// [`build_spec`](Arg::build_spec).
    pub fn unbound(&self, geometry: &Geometry) -> Arg<'_, Unlabelled> {
        Arg::unbound(self, geometry)
    }

    /// The widest line every operand can be served in along `axis`, which must label each
    /// operand's innermost dim.
    pub fn vector_size(
        &self,
        axis: Axis,
        operands: &[(&Geometry, &[Axis])],
        type_size: usize,
    ) -> usize {
        for (_, axes) in operands {
            assert_eq!(
                axes.last(),
                Some(&axis),
                "Launcher::vector_size: axis {axis:?} must label each operand's innermost dim"
            );
        }
        // A masked access counts in lines and would clip wrongly, so overhangs serve scalar.
        let overhangs = self.overhangs();
        let masked = operands
            .iter()
            .any(|(_, axes)| axes.iter().any(|a| overhangs.contains(a)));
        if masked {
            return 1;
        }
        let leaf = self.leaf_edge(axis);
        self.client
            .io_optimized_vector_sizes(type_size)
            .filter(|&v| {
                leaf.is_multiple_of(v)
                    && operands
                        .iter()
                        .all(|(g, axes)| g.serves(&[(axis, v)], axes).is_ok())
            })
            .max()
            .unwrap_or(1)
    }
}
