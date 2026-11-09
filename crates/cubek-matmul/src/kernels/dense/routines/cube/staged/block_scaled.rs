//! The settings a block-scaled product read as stored races, counted in steps of the product.
//!
//! For a prompt's rows, filled by the TMA engine through the tile DSL's transport: four by eight
//! steps on four by two planes, the box a long rhs fills the device with; two by eight, the one a
//! narrower rhs at a prompt's chunk of rows leaves fewer cores idle with; two by four on two by
//! two planes, the small box the narrowest projections take; each two stages in flight. Then four
//! by four steps on two by four planes and two by eight on four by two, three stages in flight,
//! the `128 × 128` boxes that win every Devstral Small 2 projection but its qkv at 512 and 2048
//! rows once the scales reach the instruction a word at a time; its planes store their fragments
//! ([`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine)).
//! Then four by four on two by four planes, and two by four on four by two or two by four planes,
//! three stages in flight and the box stored by the TMA engine in one bulk copy, which a
//! contraction of a few thousand finishes sooner with
//! ([`CubeStagedTmaStoredMatmulRoutine`](crate::kernels::CubeStagedTmaStoredMatmulRoutine)). A
//! stage one step deep is one tile of an input in [tile order](cubek_std::layout::tile_order),
//! which a box lands as it lands rows. Filled by the cube's units asynchronously: four by four on
//! two by four planes, three stages in flight, and two by four on four by two planes, two
//! ([`CubeStagedAsyncCopyMatmulRoutine`](crate::kernels::CubeStagedAsyncCopyMatmulRoutine)).
//!
//! For a decode's rows, one step of rows (the instruction's sixteen, every row a decode holds) by
//! one or two steps of columns, so an rhs of a few thousand columns is hundreds of boxes and every
//! core reads its share of it. Filled by the TMA engine, a stage one step deep, the most one box
//! lands, with three or four stages in flight to keep enough of the rhs in the air. Filled by the
//! units asynchronously, deeper stages: four steps on four planes side by side, or planes sharing
//! each step of `K`, which a projection with few columns and a long `K` keeps busy with, one step
//! deep on four planes where `K` divides no deeper stage.
//!
//! Every box is the one measured best for some Qwen3-4B projection on an RTX PRO 6000, the rhs
//! read back to back from memory as a model step reads it: the prompt's at 512 and 2048 rows, the
//! decode's at one and eight, in rows and in tile order.
//!
//! Apart from the float lists: a block-scaled step contracts four times the float one's `K` from
//! the same bytes, so the boxes that keep it fed are not the float product's, and the float tables
//! race none of them.

use super::base::CubeStagedMatmulRoutineBlueprint;
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulRoutine, MatmulRoutineProblem,
    MatmulView,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::{BlockScaledOperands, Space};
use cubek_microkernel::matmul::cube::{CubeStagedMicrokernelSettings, MatmulPlaneLayout};
use cubek_microkernel::matmul::plane::{PlaneFragmentMicrokernelSettings, PlaneFragmentTransport};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands, MatmulStep};
use cubek_std::layout::stored::{Operand, Packing};
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::StageRing;
use cubek_std::routine::{Grid, Routine};
use cubek_std::{Hardware, UNIT_REGISTERS, block_scaled_for, block_scaled_mma};

/// The shape of a setting a block-scaled list races, counted in steps of the product
/// ([`block_scaled_step`](CubeStagedMatmulRoutine::block_scaled_step)): a plane's steps of `M`, `N`
/// and `K` (a stage's depth), the layout of its planes, the planes sharing each step of `K` and the
/// stages in flight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StagedShape {
    pub(crate) steps: (usize, usize, usize),
    pub(crate) planes: (usize, usize),
    pub(crate) planes_along_k: usize,
    pub(crate) ring: StageRing,
}

/// Four stages in flight, each filled in its slot.
pub(crate) const QUAD: StageRing = StageRing {
    stages_in_flight: NonZeroUsize::new(4).expect("four stages"),
    ..StageRing::DOUBLE
};

/// Registers the compiler adds around a plane's lower bound
/// ([`registers_per_unit`](cubek_microkernel::matmul::plane::PlaneFragmentMicrokernel::registers_per_unit)):
/// addressing, loop state and the scale registers. A margin rather than a derivation, since the
/// compiler decides it: the four by eight box's bound of 160 compiled to 182 to 246 on
/// `sm_120`.
const ADDRESSING_REGISTERS: u32 = 64;

/// The settings of `shapes` `routine` races against a block-scaled product read as stored on `hw`,
/// each the microkernel's `in_steps` handed to `settings`: the lists of every block-scaled
/// instruction the device offers, each over inputs in its form
/// ([`block_scaled_inputs`](CubeStagedMatmulRoutine::block_scaled_inputs)), one after another, a
/// setting two instructions share listed once. A setting is counted in steps, and prepares only on
/// a product whose instruction's step it was counted in. Each is kept where it prepares on a
/// product of its own box, its stages within the device's shared memory and a unit's share of its
/// planes within [`UNIT_REGISTERS`], the compiler's addressing beside it. A device with no
/// block-scaled instruction has no step, and lists none.
pub(crate) fn block_scaled_settings<R>(
    hw: &Hardware,
    routine: R,
    shapes: &[StagedShape],
    settings: impl Fn(CubeStagedMicrokernelSettings) -> R::Settings,
) -> Vec<R::Settings>
where
    R: Routine<Problem = MatmulRoutineProblem, Blueprint = CubeStagedMatmulRoutineBlueprint>,
    R::Settings: PartialEq,
{
    let mut listed: Vec<R::Settings> = Vec::new();
    for inputs in block_scaled_mma(hw)
        .filter_map(|instruction| CubeStagedMatmulRoutine::block_scaled_inputs(&instruction.form))
    {
        let Some(step) = CubeStagedMatmulRoutine::block_scaled_step(hw, &inputs) else {
            continue;
        };
        for shape in shapes {
            let microkernel = in_steps(&step, *shape);
            let setting = settings(microkernel);
            // The product a setting is fitted on states none of the buffers the engines read, so
            // where its own engine refuses it, its box is fitted as the units fill it.
            let by_units = CubeStagedMatmulRoutineSettings::on_grid(microkernel);
            if !listed.contains(&setting)
                && fits(&step, &inputs, &microkernel, |problem| {
                    routine
                        .prepare(hw, problem, &setting)
                        .or_else(|_| CubeStagedMatmulRoutine.prepare(hw, problem, &by_units))
                        .ok()
                })
            {
                listed.push(setting);
            }
        }
    }
    listed
}

impl CubeStagedMatmulRoutine {
    /// The step of a product of `inputs` on `hw` read as stored ([`MatmulStep`]): the least box
    /// every load of both inputs fills whole — a row of the TMA engine's box, a word of scales —
    /// under the device's largest block-scaled instruction over them. `None` on a device that
    /// offers none. An input in rows and one in [tile order](cubek_std::layout::tile_order) step
    /// alike, and a load on such a device stores in this step.
    pub fn block_scaled_step(
        hw: &Hardware,
        inputs: &MatmulOperands<Option<Packing>>,
    ) -> Option<MatmulStep> {
        let (lhs, rhs) = (inputs.lhs?, inputs.rhs?);
        let instruction = block_scaled_for(
            hw,
            (&lhs.scheme, lhs.operand.scale(K)?),
            (&rhs.scheme, rhs.operand.scale(K)?),
        )
        .next()?;
        let operands = inputs.map(|_, packing| packing.map_or(Operand::plain(), |p| p.operand));
        let elements = MatmulOperands {
            lhs: instruction.shape.a_type,
            rhs: instruction.shape.b_type,
            out: instruction.shape.a_type,
        };
        let shape = Space::new(&[
            (M, instruction.shape.m as usize),
            (N, instruction.shape.n as usize),
            (K, instruction.shape.k as usize),
        ]);
        Some(
            MatmulStep::new(hw, operands, elements)
                .filled_by_tma(&operands.rhs, elements.rhs)
                .under(shape),
        )
    }

    /// Both inputs packed in `form`, their words along `K`, as a block-scaled product of that form
    /// reads them: the `[m, k]` one as a row-major quantizer writes it, the `[k, n]` one as a
    /// checkpoint packs it and as tile order stores it, each under a per-tensor `f32` above its
    /// block scales where the scales are a ratio (`ue4m3`'s) rather than a magnitude. `None` for a
    /// form no packed input stores: codes no scheme packs, or a scale type no scheme names.
    pub fn block_scaled_inputs(
        form: &BlockScaledOperands,
    ) -> Option<MatmulOperands<Option<Packing>>> {
        let scales = match form.scales {
            FloatKind::E4M3 => ScaleDtype::UE4M3,
            FloatKind::UE8M0 => ScaleDtype::UE8M0,
            _ => return None,
        };
        let code = |kind: FloatKind| match kind {
            FloatKind::E2M1x2 => Some(QuantValue::E2M1),
            FloatKind::E4M3 => Some(QuantValue::E4M3),
            FloatKind::E5M2 => Some(QuantValue::E5M2),
            _ => None,
        };
        let block = u8::try_from(form.block).ok()?;
        let scheme = |value: QuantValue, dims: &[u8], store: QuantStore| {
            let scheme = QuantScheme::default()
                .with_value(value)
                .per_block(dims, scales)
                .with_store(store);
            match scales {
                ScaleDtype::UE4M3 => scheme.per_tensor(ScaleDtype::F32),
                _ => scheme,
            }
        };
        let packing = |input: MatmulOperand, scheme: QuantScheme| {
            Some(Packing {
                operand: Operand::of_scheme(&scheme, K, input.axes())?,
                scheme,
            })
        };
        Some(MatmulOperands {
            lhs: packing(
                MatmulOperand::Lhs,
                scheme(code(form.lhs)?, &[block], QuantStore::PackedU32(0)),
            ),
            rhs: packing(
                MatmulOperand::Rhs,
                scheme(code(form.rhs)?, &[block, 1], QuantStore::PackedU32(1)),
            ),
            out: None,
        })
    }
}

/// `m` rows by `n` columns `k` deep of `inputs`, into an `f16` output: what
/// [`block_scaled_settings`] prepares its boxes on.
fn block_scaled_problem(
    inputs: &MatmulOperands<Option<Packing>>,
    m: usize,
    n: usize,
    k: usize,
) -> MatmulRoutineProblem {
    MatmulRoutineProblem::packed(m, n, k, ElemType::Float(FloatKind::F16), *inputs)
}

/// A setting of `shape`, its planes reaching their stages through the tile DSL's transport, which
/// reads the rows a box lands swizzled.
fn in_steps(step: &MatmulStep, shape: StagedShape) -> CubeStagedMicrokernelSettings {
    let calls = |count: usize, axis| {
        NonZeroUsize::new(count * step.calls(axis)).expect("a step is at least one call")
    };
    let (rows, columns, depth) = shape.steps;
    CubeStagedMicrokernelSettings {
        plane: PlaneFragmentMicrokernelSettings {
            along_m: calls(rows, M),
            along_n: calls(columns, N),
            depth: calls(depth, K),
            transport: PlaneFragmentTransport::Mma,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        },
        planes: MatmulPlaneLayout::new(shape.planes.0, shape.planes.1),
        planes_along_k: NonZeroUsize::new(shape.planes_along_k).expect("at least one plane"),
        ring: shape.ring,
        ..CubeStagedMicrokernelSettings::DEFAULT
    }
}

/// Whether `microkernel`'s settings, prepared by `prepare`, prepare a product of their own box,
/// two stages deep, and a unit's share of their planes fits its registers.
fn fits(
    step: &MatmulStep,
    inputs: &MatmulOperands<Option<Packing>>,
    microkernel: &CubeStagedMicrokernelSettings,
    prepare: impl FnOnce(&MatmulRoutineProblem) -> Option<(CubeStagedMatmulRoutineBlueprint, Grid)>,
) -> bool {
    let (plane, planes) = (&microkernel.plane, &microkernel.planes);
    let extent = |axis, fragments: NonZeroUsize, count: NonZeroUsize| {
        fragments.get() / step.calls(axis) * step.along(axis) * count.get()
    };
    let problem = block_scaled_problem(
        inputs,
        extent(M, plane.along_m, planes.along_m),
        extent(N, plane.along_n, planes.along_n),
        2 * plane.depth.get() / step.calls(K) * step.along(K),
    );
    let Some((blueprint, grid)) = prepare(&problem) else {
        return false;
    };
    let item = MatmulView::new(&problem, CubeStagedMatmulRoutine::ROWS_KIND)
        .expect(MatmulView::PREPARED)
        .work_item()
        .as_stored()
        .expect("a block-scaled product is read as stored");
    let registers = blueprint
        .microkernel
        .plane
        .registers_per_unit(&item, grid.planes().width);
    registers + ADDRESSING_REGISTERS <= UNIT_REGISTERS
}
