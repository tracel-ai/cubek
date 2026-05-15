use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

use crate::{
    components::batch::{
        BatchMatmulFamily, CheckBounds,
        gemm_plane_parallel::{
            GemmPlaneParallelBlueprint, GemmPlaneParallelFamily, KAccess, MatmulOperandLayouts,
            PlanesSplit,
        },
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine, num_concurrent_planes,
    },
};

pub struct GemmPlaneParallelRoutine {}

#[derive(Default, Clone)]
pub struct GemmPlaneParallelStrategy {
    pub target_num_planes: Option<usize>,
}

impl Display for GemmPlaneParallelStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{:?}", self.target_num_planes)
    }
}

/// Returns `(m_units, n_units)` — the count of "plane-sized" cells along
/// each output axis the kernel must cover. A staged side walks its MN
/// axis in `tile_dim`-wide slices, so the unit count is divided
/// accordingly per side.
fn output_units(
    problem: &MatmulProblem,
    kind: MatmulOperandLayouts,
    tile_dim: usize,
) -> (usize, usize) {
    let traversal = kind.k_traversal();
    let m_units = if matches!(traversal.lhs, KAccess::Staged) {
        problem.m / tile_dim
    } else {
        problem.m
    };
    let n_units = if matches!(traversal.rhs, KAccess::Staged) {
        problem.n / tile_dim
    } else {
        problem.n
    };
    (m_units, n_units)
}

/// Choose how planes within a cube enumerate the output. If exactly one
/// side is staged, planes walk that side's MN axis. Otherwise (Direct or
/// DoubleStaged) it's a free choice — defaults to N.
fn planes_split_for(kind: MatmulOperandLayouts) -> PlanesSplit {
    let traversal = kind.k_traversal();
    match (traversal.lhs, traversal.rhs) {
        (KAccess::Staged, KAccess::Direct) => PlanesSplit::M,
        _ => PlanesSplit::N,
    }
}

impl Routine<()> for GemmPlaneParallelRoutine {
    type Strategy = GemmPlaneParallelStrategy;
    type BatchMatmul = GemmPlaneParallelFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily<()>>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<()>>::Config;

    fn expand_blueprint<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let properties = device_settings.client.properties();

        match strategy {
            BlueprintStrategy::Forced(blueprint) => Ok(ExpandInfo {
                blueprint: blueprint.clone(),
                dtypes,
            }),
            BlueprintStrategy::Inferred(strategy) => {
                let target_num_planes = match strategy.target_num_planes {
                    Some(num_planes) => num_planes,
                    None => num_concurrent_planes(&properties.hardware),
                };

                let kind = MatmulOperandLayouts::from_problem(problem)?;
                let tile_dim =
                    device_settings.plane_dim as usize * device_settings.vector_sizes.rhs;
                let planes_split = planes_split_for(kind);

                // num_planes cap depends on what each plane is doing:
                // - Simple, split N: one plane per N column.
                // - Simple, split M: one plane per M row.
                // - Staged: planes share a `tile_dim` slice within a single
                //   tile (within-tile parallelism).
                let num_planes = if kind.k_traversal().is_staged() {
                    // Any staged side: planes share work within one
                    // tile_dim slice of the staged MN axis.
                    max(1, min(target_num_planes, tile_dim))
                } else {
                    // Both Direct: planes cover individual output cells.
                    match planes_split {
                        PlanesSplit::M => max(1, min(target_num_planes, problem.m)),
                        PlanesSplit::N => max(1, min(target_num_planes, problem.n)),
                    }
                };

                let (m_units, n_units) = output_units(problem, kind, tile_dim);
                let split_units = match planes_split {
                    PlanesSplit::M => m_units,
                    PlanesSplit::N => n_units,
                };
                let check_bounds = if split_units.is_multiple_of(num_planes) {
                    CheckBounds::None
                } else {
                    CheckBounds::Terminate
                };

                let blueprint = GemmPlaneParallelBlueprint {
                    dtypes: dtypes.clone(),
                    num_planes,
                    tile_dim,
                    hypercube_blueprint: HypercubeBlueprint::builder()
                        .cube_count_strategy(CubeCountStrategy::Flattened)
                        .global_order(GlobalOrder::RowMajor)
                        .build(),
                    kind,
                    planes_split,
                    check_bounds,
                };

                Ok(ExpandInfo { blueprint, dtypes })
            }
        }
    }

    fn prepare<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cube_dim = Self::BatchMatmul::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?
        .to_cube_dim(device_settings.plane_dim)?;

        let (m_units, n_units) = output_units(problem, blueprint.kind, blueprint.tile_dim);
        let (m_cubes, n_cubes) = match blueprint.planes_split {
            PlanesSplit::M => (
                m_units.div_ceil(blueprint.num_planes) as u32,
                n_units as u32,
            ),
            PlanesSplit::N => (
                m_units as u32,
                n_units.div_ceil(blueprint.num_planes) as u32,
            ),
        };

        let cube_count_plan = CubeCountPlan::from_blueprint(
            &blueprint.hypercube_blueprint,
            (m_cubes, n_cubes, problem.num_batches() as u32).into(),
            &device_settings.max_cube_count,
        );

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
        })
    }
}
