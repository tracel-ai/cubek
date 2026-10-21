use super::packed::*;
use super::problem::{MatmulProblem, OperandKind};
use super::selector::{
    FLOAT_FEW_ROWS, MatmulChoice, MatmulDequantize, MatmulPackedChoice, MatmulSelection,
    MatmulTable, NO_NVFP4_WEIGHT, select,
};
use super::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
use MatmulChoice as F;
use MatmulKernel as K;
use MatmulKernelHeuristic as H;
use MatmulOpStrategy::*;
use MatmulPackedChoice as P;
use MatmulSelection as S;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
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

/// The weight read as stored by the kernel `choice` gives.
fn packed(choice: P) -> S {
    S::Packed(choice)
}

/// The kernel a rule names, and `otherwise` where it declines.
fn rule(kernel: K, otherwise: S) -> P {
    P::Rule {
        kernel,
        otherwise: Box::new(otherwise),
    }
}

/// The weight decoded by this crate's decode, then read by the float kernel `choice` gives.
fn decoded(choice: F) -> S {
    S::DequantizeThen {
        dequantize: MatmulDequantize::Decode,
        choice,
    }
}

fn quantized(m: usize, scheme: QuantScheme) -> MatmulProblem {
    MatmulProblem {
        m,
        k: 2048,
        n: 8192,
        batches: 1,
        streamed_rows: Some(m),
        lhs: OperandKind::Float,
        rhs: OperandKind::Quantized(scheme),
    }
}

/// A quantized activation against a packed weight runs, under every strategy, as a float one
/// would once decoded; against a float weight it is refused.
#[test]
fn a_quantized_activation_is_decoded_then_runs_as_a_float_one() {
    let hw = block_scaled();
    let strategies = [
        Autotune,
        Heuristic(H::Tile),
        Heuristic(H::Cubek),
        Specific(K::UnitRegisters),
        Specific(K::PlaneShuffle),
    ];
    for rows in [1, 512] {
        for strategy in strategies {
            let float_lhs = quantized(rows, q4());
            let quantized_lhs = MatmulProblem {
                lhs: OperandKind::Quantized(nvfp4_along_k()),
                ..float_lhs
            };
            assert_eq!(
                select(strategy, &quantized_lhs, &hw),
                select(strategy, &float_lhs, &hw).map(|then| S::DecodeLhsThen(Box::new(then))),
                "{strategy:?}, {rows} rows"
            );
            let float_rhs = MatmulProblem {
                rhs: OperandKind::Float,
                ..quantized_lhs
            };
            assert!(
                select(strategy, &float_rhs, &hw).is_err(),
                "{strategy:?}, {rows} rows"
            );
        }
    }
}

/// Autotune keeps an NVFP4 weight in 16-bit whatever the device: the few-rows table at the decode
/// rows, the many-rows table past them, on a GPU with the 4-bit instruction as on one without. The
/// NVFP4 table rounds the activation, a different product, which only its pin runs.
#[test]
fn autotune_keeps_an_nvfp4_activation_in_16_bit() {
    for hw in [block_scaled(), devices::h100()] {
        for (rows, table) in [
            (1, MatmulTable::PackedFewRows),
            (QUANTIZED_FEW_ROWS, MatmulTable::PackedFewRows),
            (QUANTIZED_FEW_ROWS + 1, MatmulTable::PackedManyRows),
            (2048, MatmulTable::PackedManyRows),
        ] {
            assert_eq!(
                select(Autotune, &quantized(rows, nvfp4_along_k()), &hw),
                Ok(packed(P::Autotune(table))),
                "{rows} rows on {hw:?}"
            );
        }
    }
}

/// Named, the 4-bit tensor-core kernel runs an NVFP4 weight whose words run down `k`, on a GPU
/// with the 4-bit instruction, at any row count, and nothing else: it refuses another scheme, NVFP4 packed along `n`, and a GPU without the instruction. No policy names
/// it: a policy picks by rule and never measures, and this kernel's settings are chosen by
/// measuring.
#[test]
fn the_named_nvfp4_kernel_serves_only_an_nvfp4_weight_on_a_4_bit_gpu() {
    let named = Specific(K::Nvfp4TensorCores);
    for rows in [1, 512] {
        let nvfp4 = quantized(rows, nvfp4_along_k());
        assert_eq!(
            select(named, &nvfp4, &block_scaled()),
            Ok(packed(P::Pin(K::Nvfp4TensorCores))),
            "{rows}"
        );
        assert_eq!(
            select(named, &nvfp4, &devices::h100()),
            Err(NO_NVFP4_WEIGHT)
        );
        for refused in [quantized(rows, q4()), quantized(rows, e2m1_blocks())] {
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
                Ok(packed(P::Pin(K::Nvfp4TensorCores))),
                "{strategy:?}"
            );
        }
    }
}

/// The heuristic decode projection takes the packed plane routine at one row, and dequantizes
/// otherwise — onto this crate's float pair below the cubek ceiling, and onto cubek's own
/// selector at it, reading the weight as stored rather than materializing it.
#[test]
fn heuristic_quantized_rules() {
    let hw = devices::strix_halo();
    for (heuristic, expected) in [
        (
            H::Tile,
            rule(K::PlaneShuffle, decoded(F::Rule(K::PlaneShuffle))),
        ),
        (
            H::Cubek,
            rule(
                K::PlaneShuffle,
                packed(rule(K::CubekAuto, decoded(F::Rule(K::CubekAuto)))),
            ),
        ),
    ] {
        for scheme in [q4(), two_level()] {
            assert_eq!(
                select(Heuristic(heuristic), &quantized(1, scheme), &hw),
                Ok(packed(expected.clone())),
                "{heuristic:?}",
            );
        }
    }
    assert_eq!(
        select(Heuristic(H::Cubek), &quantized(2, q4()), &hw),
        Ok(packed(rule(K::CubekAuto, decoded(F::Rule(K::CubekAuto)))))
    );
    assert_eq!(
        select(Heuristic(H::Tile), &quantized(2, q4()), &hw),
        Ok(decoded(F::Rule(K::PlaneShuffle)))
    );
}

/// The packed plane routine binds the per-tensor scale too, as a second scale level on the
/// weight's tile.
#[test]
fn a_two_level_scheme_reaches_the_plane_routine() {
    let hw = devices::strix_halo();
    assert_eq!(
        select(Specific(K::PlaneShuffle), &quantized(1, two_level()), &hw),
        Ok(packed(P::Pin(K::PlaneShuffle)))
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
            Ok(packed(P::Autotune(MatmulTable::PackedFewRows))),
        ),
        (
            Autotune,
            quantized(512, q4()),
            Ok(packed(P::Autotune(MatmulTable::PackedManyRows))),
        ),
        (
            Specific(K::UnitRegisters),
            quantized(512, q4()),
            Ok(decoded(F::Pin(K::UnitRegisters))),
        ),
        (
            Specific(K::PlaneShuffle),
            quantized(1, q4()),
            Ok(packed(P::Pin(K::PlaneShuffle))),
        ),
        (
            Specific(K::CubekAuto),
            quantized(1, q4()),
            Ok(packed(P::Pin(K::CubekAuto))),
        ),
    ] {
        assert_eq!(select(strategy, &problem, &hw), expected, "{strategy:?}");
    }
}

/// A scheme the kernels do not stream is materialized before *every*
/// kernel, cubek's selector included.
///
/// cubek's staged routines do read a quantized operand, so handing them one
/// looks like the better trade — but `Strategy::Auto` answers a setup
/// error it cannot classify with a panic rather than an error, and a panic
/// escaping a launch leaves the stream where it was. A scheme nobody here
/// has vouched for is exactly the input that produces one.
#[test]
fn an_unsupported_scheme_never_reaches_a_quantized_kernel_at_all() {
    let hw = devices::cpu_12();
    for strategy in [
        Autotune,
        Heuristic(H::Cubek),
        Heuristic(H::Tile),
        Specific(K::UnitRegisters),
        Specific(K::PlaneShuffle),
        Specific(K::CubekAuto),
    ] {
        for rows in [0, 1, FLOAT_FEW_ROWS + 1, 512] {
            let selection = select(strategy, &quantized(rows, per_tensor()), &hw);
            assert!(
                matches!(
                    selection,
                    Ok(S::DequantizeThen {
                        dequantize: MatmulDequantize::Burn,
                        ..
                    })
                ),
                "{strategy:?} {rows} rows: {selection:?}"
            );
        }
    }
}

/// The fp4 value type reaches the packed plane routine on every device, because its decode is
/// software: `e2m1` unpacks the way the integer formats do, two codes a byte through
/// shifts and a table, and nothing about it asks the runtime to name a type. A device
/// that refused it would dequantize the default weight format on every step.
///
/// Pinned on the one hardware that used to be refused, so a decode gate that comes back
/// under another name fails here rather than routing a fleet dequantized.
#[test]
fn the_fp4_value_reaches_the_plane_routine_on_every_device() {
    let hw = devices::m2_pro();
    assert!(matmul_supported(&hw, &e2m1_blocks()));
    assert_eq!(
        select(Specific(K::PlaneShuffle), &quantized(1, e2m1_blocks()), &hw),
        Ok(packed(P::Pin(K::PlaneShuffle)))
    );
}

/// A narrow (minifloat) scale dtype is served as it is stored: the launch binds the bytes
/// as words and the kernel decodes one scale a read. `ue8m0` is MXFP4's scale level, so
/// declining it would dequantize every MXFP4 weight.
#[test]
fn a_narrow_scale_dtype_reaches_the_plane_routine() {
    let hw = devices::strix_halo();
    assert!(matmul_supported(&hw, &ue8m0_blocks()));
    assert_eq!(
        select(
            Specific(K::PlaneShuffle),
            &quantized(1, ue8m0_blocks()),
            &hw
        ),
        Ok(packed(P::Pin(K::PlaneShuffle)))
    );
}

/// The bound's edges: a packed product reaches the few-rows table from no rows to the bound.
#[test]
fn the_quantized_bound_holds_its_edges() {
    assert!(quantized_few_rows(0) && quantized_few_rows(QUANTIZED_FEW_ROWS));
    assert!(!quantized_few_rows(QUANTIZED_FEW_ROWS + 1));
}

/// A packed weight reaches its few-rows table by its own row ceiling, past the float one: the packed
/// kernels stream to [`QUANTIZED_FEW_ROWS`], a float one to [`FLOAT_FEW_ROWS`].
///
/// The two being different numbers is the property. Routing a float width past the float bound to
/// the few-rows table would put it on a table whose every candidate declines it, and the widths between
/// the bounds would be answered by the floor rather than a kernel.
#[test]
fn a_packed_weight_reaches_its_table_by_its_own_row_ceiling() {
    let hw = devices::strix_halo();
    const { assert!(FLOAT_FEW_ROWS < QUANTIZED_FEW_ROWS, "the ceilings coincide") };
    assert_eq!(
        select(Autotune, &quantized(QUANTIZED_FEW_ROWS, q4()), &hw),
        Ok(packed(P::Autotune(MatmulTable::PackedFewRows)))
    );
    assert_eq!(
        select(Autotune, &quantized(QUANTIZED_FEW_ROWS + 1, q4()), &hw),
        Ok(packed(P::Autotune(MatmulTable::PackedManyRows)))
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
        Ok(packed(rule(
            K::PlaneShuffle,
            decoded(F::Rule(K::PlaneShuffle))
        )))
    );
    for strategy in MatmulOpStrategy::all()
        .into_iter()
        .filter(MatmulOpStrategy::is_policy)
    {
        // A weight stored in tiles is the exception, and not checked here: only the packed
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
                !matches!(
                    selection,
                    Ok(S::Packed(P::Pin(_))
                        | S::DequantizeThen {
                            choice: F::Pin(_),
                            ..
                        })
                ),
                "{strategy:?} names a pin for {problem:?}: {selection:?}"
            );
        }
    }
}
