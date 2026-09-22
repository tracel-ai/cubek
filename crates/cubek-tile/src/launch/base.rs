//! One kernel launch: the [`Launcher`] binds a partitioning, the grid and the tiles its operands
//! are cut to to a client, keeping the concrete (real-extent) space beside the kernel-form one:
//! geometry and divisibility read off real extents, and nothing consumes it early.

use cubecl::ir::OpaqueType;
use cubecl::prelude::*;

use crate::{
    Arg, Axis, Geometry, Partitioning, PartitioningLaunch, Space, SpaceLaunch, Unlabelled,
};

/// How many cubes of how many units the launch runs: stated by a blueprint, or read off the
/// partitioning's levels (a test, a benchmark mapping, a kernel with no blueprint).
#[derive(Clone, Debug)]
pub enum Grid {
    Stated {
        cube_count: CubeCount,
        cube_dim: CubeDim,
    },
    FromLevels,
}

/// One launch: a partitioning (its space in kernel form, the extents a family frees stated
/// `Dynamic`), the concrete space with this call's real extents, and the grid, bound to a client.
/// Geometry and divisibility read off the concrete space, tile arguments off the kernel-form one.
///
/// A storage-tiled operand's storage tile must be the tile of one of the partitioning's levels
/// ([`Storage`](crate::Storage)); the launch is where the buffer's real extents and the levels are
/// both in hand.
#[derive(Clone)]
pub struct Launcher {
    client: Client,
    partitioning: Partitioning,
    concrete: Space,
    cube_count: CubeCount,
    cube_dim: CubeDim,
}

impl Launcher {
    /// `partitioning` over `concrete`'s real extents, on `grid`. Refuses a cube the device cannot
    /// hold, a filler role on a device with no barrier, and a lanes level that is neither one lane
    /// nor the plane when the grid is read off the levels.
    pub fn new(client: &Client, partitioning: Partitioning, concrete: &Space, grid: Grid) -> Self {
        let (cube_count, cube_dim) = match grid {
            Grid::Stated {
                cube_count,
                cube_dim,
            } => (cube_count, cube_dim),
            Grid::FromLevels => {
                let plane_size = client.properties().hardware.plane_size_max;
                let lanes = partitioning.lanes();
                assert!(
                    lanes == 1 || lanes == plane_size,
                    "Launcher: a grid read off the levels needs a lanes level of one lane or the \
                     whole plane ({plane_size}), got {lanes}"
                );
                // Counted over the real extents: the kernel-form space may state none.
                let over_concrete =
                    Partitioning::new(concrete.clone(), partitioning.levels().to_vec());
                (
                    over_concrete.cube_count(),
                    over_concrete.cube_dim(plane_size),
                )
            }
        };
        let max_units = client.properties().hardware.max_units_per_cube;
        assert!(
            cube_dim.num_elems() <= max_units,
            "Launcher: a cube of {} units, but the device holds at most {max_units}",
            cube_dim.num_elems()
        );
        let fillers = partitioning.fillers();
        // The two roles meet on a barrier and nowhere else, so a device with no barrier type would
        // run two loops with no rendezvous. Refused on the host, not where the slot is allocated: a
        // refusal at expansion fires on a worker thread, unseen, and the launch returns zeros.
        assert!(
            fillers == 0
                || client
                    .properties()
                    .features
                    .types
                    .opaque
                    .contains(&OpaqueType::Barrier),
            "Launcher: {fillers} plane(s) are set aside to fill a walk's stages, and this device \
             carries no barrier type for the two roles to meet on"
        );
        Launcher {
            client: client.clone(),
            partitioning,
            concrete: concrete.clone(),
            cube_count,
            cube_dim,
        }
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

    /// The kernel's partitioning argument: the kernel-form space, its dynamic extents sized by this
    /// launch, under the levels the launch states. What `for cube in partitioning` iterates.
    pub fn partitioning_arg(&self) -> PartitioningLaunch {
        PartitioningLaunch::new(self.space_arg(), self.partitioning.levels().to_vec())
    }

    /// The kernel-form space as a kernel argument: its shape, plus each dynamic axis's size read
    /// off the concrete space (positional over every axis, empty when the space is static).
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

    /// The partitioning's levels over the concrete space: what overhang, leaf edges and the grid
    /// are read off, since the kernel-form space may state no extent to read them from.
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

    /// Bind `binding` as an operand of this launch: the builder that derives its layout, width and
    /// bounds-check against the kernel's partitioning.
    pub fn arg(&self, binding: TensorBinding) -> Arg<'_, Unlabelled> {
        Arg::bound(self, binding)
    }

    /// [`arg`](Self::arg) over a stated geometry, for an operand with no tensor: a fused store's
    /// destination ([`Tile::of_sink`](crate::Tile::of_sink)) or a fused read's producer
    /// ([`Tile::of_source`](crate::Tile::of_source)). `geometry` is what it *would* have had.
    /// End it with [`build_spec`](Arg::build_spec), which hands back the settled geometry too.
    pub fn unbound(&self, geometry: &Geometry) -> Arg<'_, Unlabelled> {
        Arg::unbound(self, geometry)
    }

    /// The widest `Vector<E, v>` line every operand can be served in along `axis`: one width for
    /// all, since a kernel reading one operand's lines writes the other's. Takes a [`Geometry`]
    /// rather than a binding, so an operand with no tensor constrains the width like any other.
    ///
    /// `1` unless each `(geometry, axes)` is unchecked and innermost-contiguous and `v` divides
    /// each inner extent, every coarser stride and the axis's leaf tile edge.
    pub fn vector_size(
        &self,
        axis: Axis,
        operands: &[(&Geometry, &[Axis])],
        type_size: usize,
    ) -> usize {
        // The width gates below test the physical innermost dim, so `axis` must be the label
        // of every operand's innermost buffer dim (`axes` labels repeat level-major).
        for (_, axes) in operands {
            assert_eq!(
                axes.last(),
                Some(&axis),
                "Launcher::vector_size: axis {axis:?} must label each operand's innermost dim"
            );
        }
        // The one gate that is about the tiles rather than the geometry: a masked access reports
        // its length in lines and would wrongly clip, so an overhanging operand is served scalar
        // whatever its extents and strides would allow. `serves_lines` below answers the rest.
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
                    // The same gates `Geometry::serves_lines` refuses a stated width on: the
                    // innermost extent counts in lines and every coarser stride re-expresses
                    // as `stride / v`, which truncates when `v` does not divide it.
                    && operands.iter().all(|(g, _)| g.serves_lines(v).is_ok())
            })
            .max()
            .unwrap_or(1)
    }
}
