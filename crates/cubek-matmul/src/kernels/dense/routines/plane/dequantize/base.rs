use super::kernel::plane_dequantize_kernel;
use crate::kernels::tensor::{PackedTensor, PlainTensor};
use crate::kernels::{
    MatmulBlueprint, MatmulDequantizeBinding, MatmulRoutineProblem, MatmulTensors, MatmulView,
    PlaneShuffleMatmulRoutineSettings, PlaneShufflePackedPlacement,
};
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Write;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::launch::{Buffered, DestinationLaunch};
use cubecl::tile::space::CubeOrder;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulPartitioning;
use cubek_microkernel::matmul::dense::plane::PlaneShuffleMicrokernel;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// The packed weight decoded into a float weight in the activation's element, `[k, n]` in dense
/// rows of the axis its words do not run along: the first launch of a product that then runs the
/// float routine. It places the plane shuffle routine's microkernel over the weight alone, a product of one
/// row, so each unit holds whole reads of the words and whole words of the scales, and each unit
/// copies its reads into the float weight as it decodes them. Unlike that routine, it walks cubes: a
/// copy is filled by every unit of its cube, so each cube copies its planes' box.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneDequantizeMatmulRoutine;

pub type PlaneDequantizeMatmulRoutineBlueprint = MatmulBlueprint<PlaneShuffleMicrokernel>;

impl PlaneDequantizeMatmulRoutine {
    /// The settings every decode runs at: the plane shuffle routine's lead. The decode is a copy, so nothing
    /// about it is worth racing.
    pub const SETTINGS: PlaneShuffleMatmulRoutineSettings =
        PlaneShuffleMatmulRoutineSettings::PACKED;

    /// `problem`'s weight alone: the product of one row against it, which places the weight as
    /// any of its rows would.
    fn weight_of(problem: &MatmulRoutineProblem) -> MatmulRoutineProblem {
        MatmulRoutineProblem {
            stated_rows: 1,
            ..problem.clone()
        }
    }

    /// `rhs`, a packed weight in rows or in the tiles it is stored in, decoded into a float
    /// `[k, n]` weight in `lhs`'s element on `client`'s device, prepared for `hw`: the first
    /// launch of every product that then runs a float routine on the weight.
    ///
    /// # Errors
    ///
    /// Why the operands are not this kind's, and why `hw` cannot place the decode.
    pub fn launch_on(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: PlainTensor,
        rhs: PackedTensor,
        settings: &PlaneShuffleMatmulRoutineSettings,
    ) -> Result<TensorHandle, Refusal> {
        let k = rhs.extent(K);
        let lhs = MatmulTensors::beside_packed(client, lhs, [M, K], k)?;
        let problem = &MatmulRoutineProblem::packed(
            lhs.shape().num_elements() / k.max(1),
            rhs.extent(N),
            k,
            lhs.elem,
            MatmulOperands {
                rhs: Some(rhs.packing()),
                ..MatmulOperands::all(None)
            },
        );
        let (blueprint, grid) = self.prepare(hw, problem, settings)?;
        let packing = problem.packings.rhs.as_ref().expect(MatmulView::PREPARED);
        let lined = packing.operand.lined();
        // Rows of the axis the words do not run along, a line of the axis they do in each.
        let rows = match lined == K {
            true => [problem.n, problem.k],
            false => [problem.k, problem.n],
        };
        let (mut into, into_plain) = PlainTensor::empty(client, rows.to_vec().into(), lhs.elem);
        let MatmulDequantizeBinding {
            launcher,
            weight,
            out,
        } = MatmulDequantizeBinding::new(
            &rhs,
            &into_plain,
            grid.launcher(
                client,
                blueprint.partitioning.kernel(),
                blueprint.partitioning.as_tile().space(),
            ),
            blueprint.widths.rhs,
        );
        // The decode reads into the activation's element, the float weight's.
        let [_, element, _, _] = MatmulView::of::<PlaneShuffleMicrokernel>(problem)
            .expect(MatmulView::PREPARED)
            .work_item()
            .kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            plane_dequantize_kernel::launch_unchecked::<Buffered>(
                client,
                grid.launch.cube_count,
                grid.launch.cube_dim,
                out.bound_width(),
                weight,
                Buffered::arg((out, Write::Replace)),
                launcher.partitioning_arg(),
                element,
            );
        }
        // A `[n, k]` buffer read as the `[k, n]` weight it holds, `k` its contiguous axis.
        if lined == K {
            let step = into.strides()[0];
            into.metadata = Box::new(Metadata::new(vec![problem.k, problem.n], vec![1, step]));
        }
        Ok(into)
    }
}

impl Routine for PlaneDequantizeMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = PlaneShuffleMicrokernel;
    type Settings = PlaneShuffleMatmulRoutineSettings;
    type Blueprint = PlaneDequantizeMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// [`SETTINGS`](Self::SETTINGS) alone.
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        vec![Self::SETTINGS]
    }

    /// Nothing of its own: it walks cubes and planes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &PlaneShuffleMicrokernel,
        _: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each cube reads the words and scales of its columns once, and writes them decoded, a
    /// value of the activation's element each. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &PlaneDequantizeMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let packing = problem.packings.rhs.as_ref().expect(MatmulView::PREPARED);
        let weight = Self::weight_of(problem);
        let item = MatmulView::of::<PlaneShuffleMicrokernel>(&weight)
            .expect(MatmulView::PREPARED)
            .work_item();
        let columns = blueprint.partitioning.columns();
        let k = problem.k as u64;
        let read = packing.bytes(problem.k * columns as usize) as u64;
        let element = item.elements.get(MatmulOperand::Lhs).size() as u64;
        estimate.staging = Some(CubeStaging {
            operations: (columns * k) as f64,
            reads: vec![GlobalStream {
                bytes: read,
                run_bytes: (read / columns.max(1)) as usize,
            }],
            writes: vec![GlobalStream {
                bytes: columns * k * element,
                run_bytes: (k * element) as usize,
            }],
            stages: 1,
            stages_in_flight: 1,
            drain_atomics: 0,
        });
        blueprint
            .microkernel
            .estimate(&item, &(), grid.planes(), estimate);
    }

    /// The plane shuffle routine's placement of the weight alone (`PlaneShufflePackedPlacement`): the
    /// same microkernel and planes over the same space, one row, the planes in cubes.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let weight = Self::weight_of(problem);
        let PlaneShufflePackedPlacement {
            microkernel,
            columns,
            planes,
        } = PlaneShufflePackedPlacement::new(hw, &weight, settings)?;
        let packing = problem.packings.rhs.as_ref().expect(MatmulView::PREPARED);
        let partitioning = MatmulView::of::<PlaneShuffleMicrokernel>(&weight)?.partitioning(
            microkernel.tiling().planes(&[(columns, planes)]),
            &[columns],
            CubeOrder::RowMajor,
            MatmulView::ONE_CUBE_A_BOX,
        );
        let cube = CubeResources {
            plane_width: microkernel.plane_width() as u32,
            shared_memory: 0,
        };
        let blueprint = MatmulBlueprint::packed(packing, microkernel, partitioning);
        let grid = Grid::new(hw, blueprint.partitioning.as_tile(), cube)?;
        Ok((blueprint, grid))
    }
}
