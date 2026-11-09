use super::packed::*;
use super::problem::{MatmulProblem, OperandKind};
use super::selector::{BLOCK_SCALED_TENSOR_CORES_UNSERVED, measured_autotune};
use super::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
use crate::kernels::tune::{
    FLOAT_FEW_ROWS, MatmulAutotune, MatmulAutotuneGroups, QUANTIZED_FEW_ROWS, quantized_few_rows,
};
use MatmulKernel as K;
use MatmulKernelHeuristic as H;
use MatmulOpStrategy::*;
use cubecl::quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::plan::Plan;
use cubek_std::{Hardware, devices};

fn q4() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .per_block([32], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
}

/// Block scales under a per-tensor second level, at the f16 blocks that level exists for.
fn two_level() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .per_block([32], ScaleDtype::F16)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
}

/// A scheme the kernels never stream: per-tensor scales.
fn per_tensor() -> QuantScheme {
    QuantScheme::default().with_value(QuantValue::Q8S)
}

/// `e2m1` values under f16 block scales: the fp4 value type on the one scale level
/// every device stores, so what is asked is about the value alone.
fn e2m1_blocks() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([32], ScaleDtype::F16)
        .with_store(QuantStore::PackedU32(0))
}

/// NVFP4 as a model serves it: `e2m1` values under 16-wide `ue4m3` blocks and one `f32` per
/// tensor, its words and blocks running down `k`.
fn nvfp4_along_k() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([16, 1], ScaleDtype::UE4M3)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(1))
}

/// A device offering the block-scaled instruction over NVFP4, as Blackwell does.
fn block_scaled() -> Hardware {
    devices::offering_nvfp4(devices::h100())
}

/// MXFP4's scale level: `ue8m0` block scales, an exponent-only byte rather than a stored
/// float, decoded by the kernel as it reads each one.
fn ue8m0_blocks() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .per_block([32], ScaleDtype::UE8M0)
        .with_store(QuantStore::PackedU32(0))
}

/// What `strategy`'s plan for `problem` on `hw` tries, as its tree prints it.
fn select(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
) -> Result<String, Refusal> {
    super::selector::select(strategy, problem, hw, MatmulAutotuneGroups::default())
        .map(|plan| plan.tree().to_string())
}

/// The fastest candidate of `autotune`.
fn fastest(autotune: MatmulAutotune) -> String {
    format!("fastest({})", autotune.name())
}

/// `kernel` alone over float inputs: what a pin of it runs.
fn float_pin(kernel: K) -> String {
    match kernel {
        K::UnitRegisters => "unit-registers@default".to_string(),
        K::PlaneShuffle => "plane-shuffle".to_string(),
        K::BlockScaledTensorCores => unreachable!("a float product's selection refuses it"),
    }
}

/// A float rule: the plane shuffle, then the fixed routines that serve what it declines.
fn float_rule() -> String {
    "in-order[plane-shuffle, cube-staged@default, unit-registers@default]".to_string()
}

/// `kernel` alone reading the packed inputs as stored: what a pin of it runs.
fn packed_pin(kernel: K) -> String {
    match kernel {
        K::PlaneShuffle => "plane-shuffle(packed as stored)".to_string(),
        K::BlockScaledTensorCores => fastest(MatmulAutotune::BlockScaled),
        K::UnitRegisters => unreachable!("a packed selection runs it on the decoded inputs"),
    }
}

/// `kernel` reading the packed inputs as stored, and `otherwise` where it declines.
fn rule(kernel: K, otherwise: String) -> String {
    let otherwise = otherwise
        .strip_prefix("in-order[")
        .and_then(|list| list.strip_suffix(']'))
        .map(str::to_string)
        .unwrap_or(otherwise);
    format!("in-order[{}, {otherwise}]", packed_pin(kernel))
}

/// The packed inputs decoded by this crate's decode, then `plan` over them.
fn decoded(plan: String) -> String {
    format!("decoded({plan})")
}

/// Whether `plan` dequantizes its packed inputs by the framework first.
fn framework_dequantized(plan: &Result<String, Refusal>) -> bool {
    matches!(plan, Ok(tree) if tree.starts_with("framework-dequantized("))
}

/// Whether `plan` runs one kernel and nothing after it, once its steps are taken: a pin.
fn names_a_pin(plan: &str) -> bool {
    let mut inner = plan;
    for step in ["decoded(", "framework-dequantized("] {
        if let Some(rest) = inner.strip_prefix(step) {
            inner = rest;
        }
    }
    !(inner.starts_with("fastest(") || inner.starts_with("in-order["))
}

/// `m` rows against a `[k, n]` rhs packed under `scheme`, the lhs float.
fn quantized(m: usize, scheme: QuantScheme) -> MatmulProblem {
    MatmulProblem {
        m,
        k: 2048,
        n: 8192,
        batches: 1,
        streamed_rows: Some(m),
        operands: MatmulOperands {
            lhs: OperandKind::Float,
            rhs: OperandKind::Quantized(scheme),
            out: OperandKind::Float,
        },
    }
}

/// `problem` with its lhs packed under `scheme` too.
fn with_packed_lhs(problem: MatmulProblem, scheme: QuantScheme) -> MatmulProblem {
    MatmulProblem {
        operands: MatmulOperands {
            lhs: OperandKind::Quantized(scheme),
            ..problem.operands
        },
        ..problem
    }
}

/// `problem` with its rhs float.
fn with_float_rhs(problem: MatmulProblem) -> MatmulProblem {
    MatmulProblem {
        operands: MatmulOperands {
            rhs: OperandKind::Float,
            ..problem.operands
        },
        ..problem
    }
}

/// NVFP4 over an `[m, k]` lhs: its words and blocks running along `k`, its last axis.
fn nvfp4_lhs() -> QuantScheme {
    nvfp4_scheme()
}

/// Which input is packed never changes the selection: under every strategy, a packed lhs against
/// a float rhs, a float lhs against a packed rhs and both packed are selected alike.
#[test]
fn which_input_is_packed_never_changes_the_selection() {
    let hw = block_scaled();
    let strategies = [
        Autotune,
        Heuristic(H::Tile),
        Specific(K::UnitRegisters),
        Specific(K::PlaneShuffle),
    ];
    for rows in [1, 512] {
        for strategy in strategies {
            let packed_rhs = quantized(rows, q4());
            let packed_lhs = with_float_rhs(with_packed_lhs(packed_rhs, q4()));
            let both = with_packed_lhs(packed_rhs, nvfp4_lhs());
            let selection = select(strategy, &packed_rhs, &hw);
            assert_eq!(
                select(strategy, &packed_lhs, &hw),
                selection,
                "{strategy:?}, {rows} rows"
            );
            assert_eq!(
                select(strategy, &both, &hw),
                selection,
                "{strategy:?}, {rows} rows"
            );
        }
    }
}

/// The autotune a caller asks which routine won is the one its plan is: autotune's, on the autotune
/// the inputs go to, and none where the plan is anything else.
#[test]
fn the_measured_autotune_is_the_plan_where_it_is_one_autotune() {
    let hw = block_scaled();
    let measured = |strategy, problem: &MatmulProblem| measured_autotune(strategy, problem, &hw);
    assert_eq!(
        measured(Autotune, &quantized(1, q4())),
        Some(MatmulAutotune::FewRows)
    );
    assert_eq!(
        measured(Autotune, &quantized(QUANTIZED_FEW_ROWS + 1, q4())),
        Some(MatmulAutotune::ManyRows)
    );
    assert_eq!(
        measured(Autotune, &with_float_rhs(quantized(512, q4()))),
        Some(MatmulAutotune::ManyRows)
    );
    assert_eq!(measured(Autotune, &quantized(1, per_tensor())), None);
    assert_eq!(measured(Heuristic(H::Tile), &quantized(1, q4())), None);
    for problem in [quantized(1, q4()), with_float_rhs(quantized(4, q4()))] {
        let autotune = measured(Autotune, &problem).expect("autotune measures");
        assert_eq!(select(Autotune, &problem, &hw), Ok(fastest(autotune)));
    }
}

/// A packed input under a scheme no kernel reads sends the product to the framework's dequantize,
/// whichever input it is.
#[test]
fn an_unread_scheme_on_either_input_is_dequantized_by_the_framework() {
    let hw = block_scaled();
    let burn = |problem: &MatmulProblem| framework_dequantized(&select(Autotune, problem, &hw));
    let unread_rhs = quantized(8, per_tensor());
    assert!(burn(&unread_rhs));
    assert!(burn(&with_float_rhs(with_packed_lhs(
        unread_rhs,
        per_tensor()
    ))));
    assert!(burn(&with_packed_lhs(quantized(8, q4()), per_tensor())));
}

/// Autotune never sends two NVFP4 inputs to the 4-bit tensor cores, whatever the device: the
/// few-rows autotune at the decode rows, the many-rows autotune past them, on a GPU with the 4-bit
/// instruction as on one without. Only its pin runs the block-scaled autotune.
#[test]
fn autotune_never_names_the_4_bit_kernel() {
    for hw in [block_scaled(), devices::h100()] {
        for (rows, autotune) in [
            (1, MatmulAutotune::FewRows),
            (QUANTIZED_FEW_ROWS, MatmulAutotune::FewRows),
            (QUANTIZED_FEW_ROWS + 1, MatmulAutotune::ManyRows),
            (2048, MatmulAutotune::ManyRows),
        ] {
            assert_eq!(
                select(
                    Autotune,
                    &with_packed_lhs(quantized(rows, nvfp4_along_k()), nvfp4_lhs()),
                    &hw
                ),
                Ok(fastest(autotune)),
                "{rows} rows on {hw:?}"
            );
        }
    }
}

/// Named, the 4-bit tensor-core kernel runs two NVFP4 inputs whose words run down `k`, on a GPU
/// with the 4-bit instruction, at any row count, and nothing else: it refuses a float input,
/// another scheme, and a GPU without the instruction. No policy names it: a policy picks by rule
/// and never measures, and this kernel's settings are chosen by measuring.
#[test]
fn the_named_block_scaled_kernel_serves_two_nvfp4_inputs_only_on_a_gpu_with_the_instruction() {
    let named = Specific(K::BlockScaledTensorCores);
    for rows in [1, 512] {
        let nvfp4 = with_packed_lhs(quantized(rows, nvfp4_along_k()), nvfp4_lhs());
        assert_eq!(
            select(named, &nvfp4, &block_scaled()),
            Ok(packed_pin(K::BlockScaledTensorCores)),
            "{rows}"
        );
        assert_eq!(
            select(named, &nvfp4, &devices::h100()),
            Err(BLOCK_SCALED_TENSOR_CORES_UNSERVED)
        );
        for refused in [
            quantized(rows, nvfp4_along_k()),
            with_packed_lhs(quantized(rows, q4()), nvfp4_lhs()),
            with_packed_lhs(quantized(rows, e2m1_blocks()), nvfp4_lhs()),
        ] {
            assert!(
                select(named, &refused, &block_scaled()).is_err(),
                "{refused:?}"
            );
        }
        for strategy in MatmulOpStrategy::all()
            .into_iter()
            .filter(MatmulOpStrategy::is_policy)
        {
            assert_ne!(
                select(strategy, &nvfp4, &block_scaled()),
                Ok(packed_pin(K::BlockScaledTensorCores)),
                "{strategy:?}"
            );
        }
    }
}

/// Named, the block-scaled kernel takes any pair of formats the GPU's block-scaled instruction
/// reads a word of scales a step of: NVFP4 without its per-tensor scale, and `e2m1` under
/// `ue8m0` scales of sixteen. It refuses a value no instruction reads, and MXFP4's `e2m1` and
/// MXFP8's `e4m3` under scales of thirty-two, whose step of the instruction cuts a word of
/// scales, a window the tile DSL does not read.
#[test]
fn the_named_block_scaled_kernel_serves_each_format_its_instruction_reads() {
    let named = Specific(K::BlockScaledTensorCores);
    let hw = devices::offering_block_scaled(devices::h100());
    let scheme = |value, block: u8, scale, store| {
        let dims: &[u8] = match store {
            QuantStore::PackedU32(1) => &[block, 1],
            _ => &[block],
        };
        QuantScheme::default()
            .with_value(value)
            .per_block(dims, scale)
            .with_store(store)
    };
    let pair = |value, block, scale| {
        with_packed_lhs(
            quantized(512, scheme(value, block, scale, QuantStore::PackedU32(1))),
            scheme(value, block, scale, QuantStore::PackedU32(0)),
        )
    };
    let served = [
        pair(QuantValue::E2M1, 16, ScaleDtype::UE4M3),
        pair(QuantValue::E2M1, 16, ScaleDtype::UE8M0),
    ];
    for problem in served {
        assert_eq!(
            select(named, &problem, &hw),
            Ok(packed_pin(K::BlockScaledTensorCores)),
            "{problem:?}"
        );
    }
    let refused = [
        pair(QuantValue::Q4S, 16, ScaleDtype::UE4M3),
        pair(QuantValue::E4M3, 16, ScaleDtype::UE8M0),
        pair(QuantValue::E2M1, 32, ScaleDtype::UE8M0),
        pair(QuantValue::E4M3, 32, ScaleDtype::UE8M0),
    ];
    for problem in refused {
        assert_eq!(
            select(named, &problem, &hw),
            Err(BLOCK_SCALED_TENSOR_CORES_UNSERVED),
            "{problem:?}"
        );
    }
}

/// The heuristic decode projection takes the packed plane routine at one row, and dequantizes
/// otherwise, onto this crate's float routines.
#[test]
fn heuristic_quantized_rules() {
    let hw = devices::strix_halo();
    for scheme in [q4(), two_level()] {
        assert_eq!(
            select(Heuristic(H::Tile), &quantized(1, scheme), &hw),
            Ok(rule(K::PlaneShuffle, decoded(float_rule()))),
        );
    }
    assert_eq!(
        select(Heuristic(H::Tile), &quantized(2, q4()), &hw),
        Ok(decoded(float_rule()))
    );
}

/// The packed plane routine binds the per-tensor scale too, as a second scale level on the
/// packed input's tile.
#[test]
fn a_two_level_scheme_reaches_the_plane_routine() {
    let hw = devices::strix_halo();
    assert_eq!(
        select(Specific(K::PlaneShuffle), &quantized(1, two_level()), &hw),
        Ok(packed_pin(K::PlaneShuffle))
    );
}

/// The autotuned and pinned quantized rows.
#[test]
fn autotuned_and_pinned_quantized_rules() {
    let hw = devices::strix_halo();
    for (strategy, problem, expected) in [
        (
            Autotune,
            quantized(1, q4()),
            Ok(fastest(MatmulAutotune::FewRows)),
        ),
        (
            Autotune,
            quantized(512, q4()),
            Ok(fastest(MatmulAutotune::ManyRows)),
        ),
        (
            Specific(K::UnitRegisters),
            quantized(512, q4()),
            Ok(decoded(float_pin(K::UnitRegisters))),
        ),
        (
            Specific(K::PlaneShuffle),
            quantized(1, q4()),
            Ok(packed_pin(K::PlaneShuffle)),
        ),
    ] {
        assert_eq!(select(strategy, &problem, &hw), expected, "{strategy:?}");
    }
}

/// A scheme the kernels do not stream is materialized before *every* kernel: a scheme nobody here
/// has vouched for is read by no kernel as stored.
#[test]
fn an_unsupported_scheme_never_reaches_a_quantized_kernel_at_all() {
    let hw = devices::cpu_12();
    for strategy in [
        Autotune,
        Heuristic(H::Tile),
        Specific(K::UnitRegisters),
        Specific(K::PlaneShuffle),
    ] {
        for rows in [0, 1, FLOAT_FEW_ROWS + 1, 512] {
            let selection = select(strategy, &quantized(rows, per_tensor()), &hw);
            assert!(
                framework_dequantized(&selection),
                "{strategy:?} {rows} rows: {selection:?}"
            );
        }
    }
}

/// A lookup scheme's codes are indices into a table no packed input binds: read as stored they
/// would be the integers they are written as, so the framework decodes them first, whatever
/// value type names their width.
#[test]
fn a_lookup_scheme_never_reaches_a_quantized_kernel() {
    let hw = devices::m2_pro();
    for value in [QuantValue::Q4S, QuantValue::Q8S, QuantValue::E2M1] {
        let lookup = QuantScheme::default()
            .with_mode(QuantMode::Lookup)
            .with_value(value)
            .per_block([32], ScaleDtype::F16)
            .with_store(QuantStore::PackedU32(0));
        assert!(
            !matmul_supported(&hw, &lookup, MatmulOperand::Rhs),
            "{value:?}"
        );
        for strategy in [Autotune, Specific(K::PlaneShuffle)] {
            let selection = select(strategy, &quantized(1, lookup), &hw);
            assert!(
                framework_dequantized(&selection),
                "{value:?} {strategy:?}: {selection:?}"
            );
        }
    }
}

/// The fp4 value type reaches the packed plane routine on every device, because its decode is
/// software: `e2m1` unpacks the way the integer formats do, two codes a byte through
/// shifts and a table, and nothing about it asks the runtime to name a type. A device
/// that refused it would dequantize the default packed format on every step.
///
/// Pinned on the one hardware that used to be refused, so a decode gate that comes back
/// under another name fails here rather than routing a fleet dequantized.
#[test]
fn the_fp4_value_reaches_the_plane_routine_on_every_device() {
    let hw = devices::m2_pro();
    assert!(matmul_supported(&hw, &e2m1_blocks(), MatmulOperand::Rhs));
    assert_eq!(
        select(Specific(K::PlaneShuffle), &quantized(1, e2m1_blocks()), &hw),
        Ok(packed_pin(K::PlaneShuffle))
    );
}

/// The fp8 value types reach the packed plane routine on every device, for the reason
/// `e2m1`'s does: a code is a byte of a word, decoded in software. MXFP8 is `e4m3` under
/// `ue8m0` blocks, so a device that refused it would dequantize every MXFP8 input.
#[test]
fn the_fp8_values_reach_the_plane_routine_on_every_device() {
    let hw = devices::m2_pro();
    for value in [QuantValue::E4M3, QuantValue::E5M2] {
        let mxfp8 = QuantScheme::default()
            .with_value(value)
            .per_block([32], ScaleDtype::UE8M0)
            .with_store(QuantStore::PackedU32(0));
        assert!(
            matmul_supported(&hw, &mxfp8, MatmulOperand::Rhs),
            "{value:?}"
        );
        assert_eq!(
            select(Specific(K::PlaneShuffle), &quantized(1, mxfp8), &hw),
            Ok(packed_pin(K::PlaneShuffle)),
            "{value:?}"
        );
    }
}

/// A narrow (minifloat) scale dtype is served as it is stored: the launch binds the bytes
/// as words and the kernel decodes one scale a read. `ue8m0` is MXFP4's scale level, so
/// declining it would dequantize every MXFP4 input.
#[test]
fn a_narrow_scale_dtype_reaches_the_plane_routine() {
    let hw = devices::strix_halo();
    assert!(matmul_supported(&hw, &ue8m0_blocks(), MatmulOperand::Rhs));
    assert_eq!(
        select(
            Specific(K::PlaneShuffle),
            &quantized(1, ue8m0_blocks()),
            &hw
        ),
        Ok(packed_pin(K::PlaneShuffle))
    );
}

/// The bound's edges: a packed product reaches the few-rows autotune from no rows to the bound.
#[test]
fn the_quantized_bound_holds_its_edges() {
    assert!(quantized_few_rows(0) && quantized_few_rows(QUANTIZED_FEW_ROWS));
    assert!(!quantized_few_rows(QUANTIZED_FEW_ROWS + 1));
}

/// A packed input reaches its few-rows autotune by its own row ceiling, past the float one: the packed
/// kernels stream to [`QUANTIZED_FEW_ROWS`], a float one to [`FLOAT_FEW_ROWS`].
///
/// The two being different numbers is the property. Routing a float width past the float bound to
/// the few-rows autotune would put it on an autotune whose every candidate declines it, and the widths between
/// the bounds would be answered by the floor rather than a kernel.
#[test]
fn a_packed_input_reaches_its_autotune_by_its_own_row_ceiling() {
    let hw = devices::strix_halo();
    const { assert!(FLOAT_FEW_ROWS < QUANTIZED_FEW_ROWS, "the ceilings coincide") };
    assert_eq!(
        select(Autotune, &quantized(QUANTIZED_FEW_ROWS, q4()), &hw),
        Ok(fastest(MatmulAutotune::FewRows))
    );
    assert_eq!(
        select(Autotune, &quantized(QUANTIZED_FEW_ROWS + 1, q4()), &hw),
        Ok(fastest(MatmulAutotune::ManyRows))
    );
}

/// A policy never names a kernel that may decline without a fallback beside it: the launches
/// derive geometry, read strides, and ask cubek's selector for a plan, none of which these rules
/// can see, and a pin panics on a decline.
#[test]
fn a_policy_never_names_a_kernel_that_can_decline() {
    let hw = devices::strix_halo();
    let awkward = MatmulProblem {
        n: 40,
        ..quantized(1, q4())
    };
    assert_eq!(
        select(Heuristic(H::Tile), &awkward, &hw),
        Ok(rule(K::PlaneShuffle, decoded(float_rule())))
    );
    for strategy in MatmulOpStrategy::all()
        .into_iter()
        .filter(MatmulOpStrategy::is_policy)
    {
        // An input stored in tiles is the exception, and not checked here: only the packed
        // routines read one, so they are what every strategy runs. A pin's kernel
        // declines on what these rules cannot see: the plane shuffle on strides, cubek's selector on
        // "no accelerated path on this device" at the launch, the packed plane routine on a
        // contraction its steps do not close. A policy names a kernel with a fallback, never a pin.
        for problem in [
            quantized(1, q4()),
            quantized(4, q4()),
            quantized(1, per_tensor()),
            awkward,
        ] {
            let selection = select(strategy, &problem, &hw);
            assert!(
                !selection.as_deref().is_ok_and(names_a_pin),
                "{strategy:?} names a pin for {problem:?}: {selection:?}"
            );
        }
    }
}

mod strategy {
    use super::super::*;

    /// Everything that is not a pin lets the caller choose; a `Specific`
    /// names the kernel a benchmark row or a capture asked for and must not be
    /// substituted.
    #[test]
    fn the_policies_are_the_strategies_that_name_no_kernel() {
        let policies: Vec<_> = MatmulOpStrategy::all()
            .into_iter()
            .filter(MatmulOpStrategy::is_policy)
            .collect();
        assert_eq!(
            policies,
            vec![
                MatmulOpStrategy::Autotune,
                MatmulOpStrategy::Heuristic(MatmulKernelHeuristic::Tile),
            ],
        );
    }

    /// Autotune is what a caller naming nothing gets.
    #[test]
    fn the_default_is_autotune() {
        assert_eq!(MatmulOpStrategy::default(), MatmulOpStrategy::Autotune);
    }

    /// No pin names a framework op. The autotunes are built from kernels, and the
    /// pin vocabulary is what a benchmark drives one of them with — a variant
    /// reaching burn's `matmul` would be a candidate whose winner is another
    /// tuner's answer.
    #[test]
    fn every_pin_names_a_kernel() {
        for strategy in MatmulOpStrategy::all() {
            let MatmulOpStrategy::Specific(kernel) = strategy else {
                continue;
            };
            assert!(
                matches!(
                    kernel,
                    MatmulKernel::UnitRegisters
                        | MatmulKernel::PlaneShuffle
                        | MatmulKernel::BlockScaledTensorCores
                ),
                "{kernel:?} is not one of this family's kernels",
            );
        }
    }
}

mod float {
    use super::super::problem::*;
    use super::super::selector::*;
    use super::super::*;
    use super::{fastest, float_pin, float_rule};
    use crate::kernels::tune::{FLOAT_FEW_ROWS, MatmulAutotune, float_few_rows};
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    use cubecl::tile::launch::Refusal;
    use cubek_microkernel::matmul::MatmulOperands;
    use cubek_std::devices;

    /// The kernel `strategy` chooses for the float `problem`: the selection, which reads a float
    /// input on any device alike.
    fn select(strategy: MatmulOpStrategy, problem: &MatmulProblem) -> Result<String, Refusal> {
        super::select(strategy, problem, &devices::cpu_12())
    }

    /// A product against an unbatched rhs, so the left-hand side's extents
    /// merge: `m` rows in each of `batches` of them is what one memory-bound
    /// launch streams.
    fn float(m: usize, batches: usize) -> MatmulProblem {
        MatmulProblem {
            m,
            k: 2048,
            n: 8192,
            batches,
            streamed_rows: Some(m * batches),
            operands: MatmulOperands::all(OperandKind::Float),
        }
    }

    /// The same shape against an rhs of its own per batch — nothing merges,
    /// and no memory-bound kernel here reads more than one rhs.
    fn batched_rhs(m: usize, batches: usize) -> MatmulProblem {
        MatmulProblem {
            streamed_rows: None,
            ..float(m, batches)
        }
    }

    /// Every row of the float autotune: a float product never asks the hardware anything, so its
    /// selection is asked of no device.
    #[test]
    fn float_rules() {
        for (strategy, problem, expected) in [
            // Rows pick the autotune, and nothing else does.
            (Autotune, float(1, 1), Ok(fastest(MatmulAutotune::FewRows))),
            (
                Autotune,
                float(512, 1),
                Ok(fastest(MatmulAutotune::ManyRows)),
            ),
            // No heuristic measures: the answer is this crate's routines,
            // resolved at the launch by the strides.
            (Heuristic(H::Tile), float(1, 1), Ok(float_rule())),
            (Heuristic(H::Tile), float(512, 4), Ok(float_rule())),
            (
                Specific(K::UnitRegisters),
                float(512, 1),
                Ok(float_pin(K::UnitRegisters)),
            ),
            (
                Specific(K::PlaneShuffle),
                float(1, 1),
                Ok(float_pin(K::PlaneShuffle)),
            ),
        ] {
            assert_eq!(select(strategy, &problem), expected, "{strategy:?}");
        }
        // The 4-bit kernel reads a packed input: there is no float product for it to serve, at
        // any row count.
        for m in [1, 2, 512] {
            assert_eq!(
                select(Specific(K::BlockScaledTensorCores), &float(m, 1)),
                Err(BLOCK_SCALED_TENSOR_CORES_UNSERVED)
            );
        }
    }

    /// The float bound is what every float few-rows candidate holds: the plane shuffle compiles at most
    /// that many rows in, so a wider float width routed to the few-rows autotune would meet a candidate
    /// that declines it and be answered by the fallback.
    #[test]
    fn the_float_bound_is_what_the_plane_shuffle_holds() {
        use cubek_microkernel::matmul::plane::PlaneShuffleMicrokernel;
        assert_eq!(FLOAT_FEW_ROWS, PlaneShuffleMicrokernel::MAX_ROWS);
    }

    /// The bound's edges: a float product reaches the few-rows autotune from one row to the bound.
    #[test]
    fn the_float_bound_holds_its_edges() {
        assert!(!float_few_rows(0));
        assert!(float_few_rows(1) && float_few_rows(FLOAT_FEW_ROWS));
        assert!(!float_few_rows(FLOAT_FEW_ROWS + 1));
    }

    /// A row count picks the autotune, and nothing else does: no rhs layout narrows what reaches
    /// the few-rows autotune.
    #[test]
    fn the_row_count_alone_picks_the_autotune() {
        for rows in 1..=FLOAT_FEW_ROWS {
            assert_eq!(
                select(Autotune, &float(rows, 1)),
                Ok(fastest(MatmulAutotune::FewRows)),
                "{rows} float rows"
            );
        }
        for rows in [0, FLOAT_FEW_ROWS + 1, 512] {
            assert_eq!(
                select(Autotune, &float(rows, 1)),
                Ok(fastest(MatmulAutotune::ManyRows)),
                "{rows} float rows"
            );
        }
    }

    /// Routing counts the rows a memory-bound kernel would stream, not the
    /// trailing extent.
    ///
    /// `[b, m, k] @ [1, k, n]` is one product of `b·m` rows to every memory-bound
    /// candidate — they all merge the leading extents and read the rhs once.
    /// Routing on `m` alone calls a 512-row product an 8-row one and sends it
    /// to an autotune on which the float kernel rejects the plane shuffle, cubek's vec-mats
    /// reject anything past one row, and the answer is whatever is left. This
    /// reaches the attention score product and every backward.
    #[test]
    fn a_merged_batch_is_routed_by_the_rows_a_memory_bound_kernel_would_stream() {
        let merged = float(8, 64);
        assert_eq!(
            merged.m, 8,
            "the trailing extent is inside the memory-bound regime"
        );
        assert_eq!(merged.streamed_rows, Some(512));
        assert_eq!(
            select(Autotune, &merged),
            Ok(fastest(MatmulAutotune::ManyRows))
        );

        // An rhs per batch merges nothing, at any row count.
        for problem in [batched_rhs(1, 64), batched_rhs(8, 64)] {
            assert_eq!(
                select(Autotune, &problem),
                Ok(fastest(MatmulAutotune::ManyRows)),
                "{problem:?}"
            );
        }
    }

    /// A policy never names a kernel that may decline without a fallback beside it: the launches
    /// read strides and ask cubek's selector for a plan, which these rules cannot see, and a pin
    /// panics on a decline.
    #[test]
    fn a_policy_never_names_a_pin() {
        for strategy in MatmulOpStrategy::all()
            .into_iter()
            .filter(MatmulOpStrategy::is_policy)
        {
            for problem in [float(1, 1), float(512, 1)] {
                let selection = select(strategy, &problem);
                assert!(
                    !selection.as_deref().is_ok_and(super::names_a_pin),
                    "{strategy:?} names a pin for {problem:?}: {selection:?}"
                );
            }
        }
    }
}
