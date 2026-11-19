//! The tune candidate the many-rows and few-rows autotunes race a float routine as.

use crate::kernels::plan::MatmulCall;
use crate::kernels::tensor::PlainTensor;
use crate::kernels::tune::key::{CubeBlock, MaskedCells, MatmulTuneKey};
use crate::kernels::tune::{ListedFor, MatmulProduct};
use crate::kernels::{MatmulBlueprint, MatmulRoutine, MatmulRoutineProblem, MatmulTensors};
use cubecl::CubeDim;
use cubecl::ir::{ElemType, FloatKind};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::autotune::{CandidatePriority, TuneCandidate};
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::routine::{Routine, RoutineCandidate};
use cubek_std::{Hardware, Instructions, manual_mma};

/// One candidate of a matmul autotune's autotune: a float routine at one of its settings,
/// prepared for the device the autotune is built for.
///
/// Its priority is the matmul family's one rule, the output block one cube covers against the
/// key's extents ([`MatmulShape::priority`](crate::kernels::tune::key::MatmulShape::priority)).
/// These routines snap no cut to a divisor of the problem (a cube overhanging it is masked), so
/// the prepared launch's cube box is the block a setting asks for.
#[derive(Debug, Clone)]
pub struct MatmulRoutineCandidate<R: Routine> {
    candidate: RoutineCandidate<R>,
}

impl<R: Routine + Copy> MatmulRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        RoutineCandidate::list(routine, hw)
            .into_iter()
            .map(|candidate| Self { candidate })
            .collect()
    }

    /// The setting this candidate launches the routine at.
    pub fn settings(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.candidate.routine()
    }

    /// `call`'s product by this candidate's routine at its setting, or the reason it did not run:
    /// into its destination or into an output the launch allocates, on the device the candidate was
    /// listed for, each problem prepared once ([`PreparedLaunches`]).
    ///
    /// # Errors
    ///
    /// A packed input, which a float routine's settings are not listed for, what the tensors
    /// decline ([`MatmulTensors::writing_into`], [`MatmulTensors::new`]), and why the routine
    /// cannot run the product at this setting.
    pub fn launch<M: MatmulMicrokernel, SM>(
        &self,
        call: MatmulCall,
    ) -> Result<MatmulProduct, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        ListedFor::Float
            .admits(call.product())
            .map_err(|refusal| format!("{refusal:?}"))?;
        let settings = self.settings();
        let destination = call.destination;
        let (lhs, rhs) = (call.lhs.tensor, call.rhs.tensor);
        let tensors = match &destination {
            Some(destination) => MatmulTensors::writing_into(
                &call.client,
                lhs,
                rhs,
                PlainTensor::of_handle(destination.clone()),
                R::ROWS_KIND,
            ),
            None => MatmulTensors::new(&call.client, lhs, rhs, R::ROWS_KIND),
        };
        let launched = tensors.and_then(|tensors| {
            let (blueprint, grid) = self.candidate.prepare(&tensors.problem())?;
            self.routine()
                .dispatch_into_buffer(&tensors, blueprint, grid);
            Ok(match destination {
                Some(destination) => destination,
                None => tensors
                    .into_out()
                    .expect("a launch allocating its output hands it back"),
            })
        });
        launched
            .map(|handle| MatmulProduct {
                client: call.client,
                handle,
            })
            .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
    }
}

impl<R, M: MatmulMicrokernel, SM> MatmulRoutineCandidate<R>
where
    R: Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>,
{
    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// key names a packed input, which a float routine does not read, or the routine declines the
    /// problem on this device.
    fn prepared(&self, key: &MatmulTuneKey) -> Option<(MatmulBlueprint<M, SM>, CubeDim)> {
        match key.is_packed() {
            true => None,
            false => self.prepared_float(key),
        }
    }

    /// The launch this candidate prepares for the float product of `key`'s extents, whatever its
    /// packing: what a packed key's product becomes once its inputs are decoded.
    pub fn prepared_float(&self, key: &MatmulTuneKey) -> Option<(MatmulBlueprint<M, SM>, CubeDim)> {
        let problem = MatmulRoutineProblem::from(&key.shape());
        self.candidate.launch_of(&problem)
    }

    /// Where the output block one cube of `prepared` covers places it on `key`'s extents on `hw`;
    /// last where nothing is prepared, which the launch then declines before compiling anything.
    pub fn ranked(
        hw: &Hardware,
        key: &MatmulTuneKey,
        prepared: Option<(MatmulBlueprint<M, SM>, CubeDim)>,
    ) -> CandidatePriority {
        let Some((blueprint, _)) = prepared else {
            return CandidatePriority::LAST;
        };
        let cube = blueprint.partitioning.cube_space().space;
        // The priority reads the output block alone; the cube's `K` stands in for the step it does
        // not read: the whole of it, or a cube's run of it where several share a box.
        key.shape().priority(
            CubeBlock {
                m: cube.extent(M),
                n: cube.extent(N),
                k: cube.extent(K),
            },
            masked_cells(hw, key),
        )
    }
}

impl<R, M: MatmulMicrokernel, SM: PartialEq> TuneCandidate for MatmulRoutineCandidate<R>
where
    R: Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>,
{
    type Strategy = R::Settings;
    type Key = MatmulTuneKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (R::Blueprint, CubeDim);

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// Where the output block one cube of the prepared launch covers places it on the key's
    /// extents; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        Self::ranked(self.candidate.hardware(), key, self.prepared(key))
    }

    fn blueprint(&self, key: &MatmulTuneKey) -> Option<Self::Blueprint> {
        self.prepared(key)
    }
}

/// What the cells a masked grid covers cost a product of `key` on `hw`: free where the device
/// multiplies the key's element on a tensor core, accumulating at `f32`, and a cost otherwise.
fn masked_cells(hw: &Hardware, key: &MatmulTuneKey) -> MaskedCells {
    let elem = key.shape().elem;
    let f32 = ElemType::Float(FloatKind::F32);
    let matmul = &hw.properties.features.matmul;
    let tensor_core = manual_mma(hw)
        .chain(matmul.cmma.iter().copied())
        .for_triple(elem, elem, f32)
        .next()
        .is_some();
    match tensor_core {
        true => MaskedCells::Free,
        false => MaskedCells::Cost,
    }
}
