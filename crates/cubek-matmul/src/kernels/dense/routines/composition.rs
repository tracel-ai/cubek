//! Which routine serves which product, written down: every routine, at every setting any of its
//! lists races, against every pairing of an activation and a weight in each format the kernels
//! store, on each kind of device, at a prompt's rows and at one. A line names the routines that
//! prepare the product, with how many of their settings do, and for each routine none of whose
//! settings does, the refusals it gave.
//!
//! The file is the record a change to what composes is held to: a refusal that names no limit of
//! the device or of the copy engine is a composition to fix, and a line that moves is a product
//! that a routine now serves or no longer does. Rewrite it with `CUBEK_WRITE_PLANS=1` and read the
//! diff.

use crate::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine,
    CubeStagedTmaFillerMatmulRoutine, CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine,
    CubeStagedWgmmaMatmulRoutine, HypercubeLastArrivalMatmulRoutine, MatmulRoutineProblem,
    PlaneDequantizeMatmulRoutine, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
    UnitWholeKMatmulRoutine,
};
use crate::tiled::storage::{StoredValues, stored_tile, tile_order_on};
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands, MatmulStoredTile};
use cubek_std::layout::stored::{Operand, Packing};
use cubek_std::matmul_dimensions::{K, M, N};
use cubek_std::routine::Routine;
use cubek_std::{Hardware, devices};
use std::collections::BTreeMap;

/// Every product on every device is served as `composition.txt` records.
#[test]
fn every_composition_is_the_one_written_down() {
    let written = render_all();
    if std::env::var_os("CUBEK_WRITE_PLANS").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/kernels/dense/routines/composition.txt"
        );
        std::fs::write(path, &written).expect("composition.txt is writable");
        return;
    }
    assert_eq!(
        written,
        include_str!("composition.txt"),
        "a composition changed; rewrite composition.txt with CUBEK_WRITE_PLANS=1 and review the diff"
    );
}

const F16: ElemType = ElemType::Float(FloatKind::F16);
const BF16: ElemType = ElemType::Float(FloatKind::BF16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The product's extents: a 27B projection's `k` and `n`, small enough to prepare quickly and
/// whole multiples of every step and tile.
const K_EXTENT: usize = 5120;
const N_EXTENT: usize = 4096;

/// The rows a decode step and a prompt stream.
const ROWS: [usize; 2] = [1, 512];

/// How an input is stored: float elements, or codes under a scheme.
#[derive(Clone, Copy)]
enum Format {
    Float(ElemType),
    Codes(&'static str, QuantValue, u8, ScaleDtype, bool),
}

impl Format {
    fn name(self) -> String {
        match self {
            Format::Float(element) => format!("{element}"),
            Format::Codes(name, ..) => name.to_string(),
        }
    }

    /// Its scheme over an input packed along `K`, `lhs` `[m, k]` or rhs `[k, n]`.
    fn scheme(self, input: MatmulOperand) -> Option<QuantScheme> {
        let Format::Codes(_, value, block, scale, global) = self else {
            return None;
        };
        let (block, store) = match input {
            MatmulOperand::Lhs => (vec![1, block], QuantStore::PackedU32(0)),
            _ => (vec![block, 1], QuantStore::PackedU32(1)),
        };
        let scheme = QuantScheme::default()
            .with_value(value)
            .per_block(block, scale)
            .with_store(store);
        Some(match global {
            true => scheme.per_tensor(ScaleDtype::F32),
            false => scheme,
        })
    }
}

/// The formats the kernels store codes in.
fn code_formats() -> [Format; 5] {
    [
        Format::Codes("q8", QuantValue::Q8S, 32, ScaleDtype::F32, false),
        Format::Codes("q4", QuantValue::Q4S, 32, ScaleDtype::F32, false),
        Format::Codes("nvfp4", QuantValue::E2M1, 16, ScaleDtype::UE4M3, true),
        Format::Codes("mxfp4", QuantValue::E2M1, 32, ScaleDtype::UE8M0, false),
        Format::Codes("mxfp8", QuantValue::E4M3, 32, ScaleDtype::UE8M0, false),
    ]
}

/// Every pairing of an activation and a weight: a float activation against a weight of its own
/// element or in any code format, and an activation in any code format against a weight in any.
fn pairings() -> Vec<(Format, Format)> {
    let mut pairs = Vec::new();
    for element in [F16, BF16] {
        let float = Format::Float(element);
        pairs.push((float, float));
        pairs.extend(code_formats().map(|weight| (float, weight)));
    }
    for activation in code_formats() {
        pairs.extend(code_formats().map(|weight| (activation, weight)));
    }
    pairs
}

/// `16-bit` floats of `shape`, summed in `f32`.
fn sixteen_bit(m: u32, n: u32, k: u32) -> impl Iterator<Item = MmaConfig> {
    [F16, BF16].into_iter().map(move |element| MmaConfig {
        a_type: element,
        b_type: element,
        cd_type: F32,
        m,
        n,
        k,
    })
}

/// The devices whose instructions differ, each offering what its vendor's backend registers: a
/// Blackwell workstation GPU with the block-scaled instruction, a Hopper datacenter GPU, an RDNA4
/// GPU under ROCm, an AMD GPU under Vulkan, whose plane width the driver picks between 32 and 64,
/// and a Mac's `8 × 8 × 8` simdgroup matrix.
fn machines() -> Vec<(&'static str, Hardware)> {
    let nvidia = |hw| {
        devices::offering_mma(
            devices::offering_cmma(hw, sixteen_bit(16, 16, 16)),
            sixteen_bit(16, 8, 16),
        )
    };
    vec![
        (
            "rtx-pro-6000",
            devices::offering_block_scaled(nvidia(devices::rtx_pro_6000())),
        ),
        ("h100", nvidia(devices::h100())),
        ("rdna4-rocm", rocm_rdna4()),
        (
            "rdna-vulkan",
            devices::offering_cmma(devices::rdna_range(), sixteen_bit(16, 16, 16)),
        ),
        (
            "m2-pro",
            devices::offering_cmma(devices::m2_pro(), sixteen_bit(8, 8, 8)),
        ),
    ]
}

/// An RDNA4 GPU as ROCm reports it: planes of the warp size, 32, and the `16 × 16 × 16` 16-bit
/// matrix cores both as the fragment API and as the manual mma.
fn rocm_rdna4() -> Hardware {
    let mut hw = devices::rdna_range();
    let topology = devices::topology(&mut hw);
    topology.plane_size_min = 32;
    topology.plane_size_max = 32;
    devices::offering_mma(
        devices::offering_cmma(hw, sixteen_bit(16, 16, 16)),
        sixteen_bit(16, 16, 16),
    )
}

/// `rows` rows of `activation` against a `K_EXTENT × N_EXTENT` weight in `weight`, the weight
/// stored as a load on `hw` stores it: in the device's tile order where it has a tile for the
/// weight's scheme, in rows otherwise. The second value says which.
fn product(
    hw: &Hardware,
    rows: usize,
    activation: Format,
    weight: Format,
) -> (MatmulRoutineProblem, &'static str) {
    let element = match activation {
        Format::Float(element) => element,
        Format::Codes(..) => F16,
    };
    let packing = |format: Format, input: MatmulOperand, dimensions| {
        let scheme = format.scheme(input)?;
        let operand = Operand::of_scheme(&scheme, K, dimensions)?;
        Some(Packing { operand, scheme })
    };
    let lhs = packing(activation, MatmulOperand::Lhs, [M, K]);
    let rhs = packing(weight, MatmulOperand::Rhs, [K, N]);
    let tiled = rhs.and_then(|rhs| {
        let tile = stored_tile(
            hw,
            MatmulOperand::Rhs,
            StoredValues::Packed(rhs.scheme),
            element,
        )?;
        let (tile_k, tile_n) = (tile.extent(K), tile.extent(N));
        let order = tile_order_on(hw, &rhs.scheme, (tile_k, tile_n), (K_EXTENT, N_EXTENT))?;
        let operand =
            rhs.operand
                .in_tile_order(order, N, hw.load_values(ElemType::UInt(UIntKind::U32)))?;
        Some((
            Packing { operand, ..rhs },
            MatmulStoredTile {
                rows: tile_n,
                columns: tile_k,
            },
        ))
    });
    let problem = match (lhs, rhs) {
        (None, None) => MatmulRoutineProblem::dense(rows, N_EXTENT, K_EXTENT, element),
        _ => MatmulRoutineProblem::packed(
            rows,
            N_EXTENT,
            K_EXTENT,
            element,
            MatmulOperands {
                lhs,
                rhs: tiled.map(|(rhs, _)| rhs).or(rhs),
                out: None,
            },
        ),
    };
    match tiled {
        Some((_, tile)) => (
            MatmulRoutineProblem {
                stored_tiles: MatmulOperands {
                    rhs: Some(tile),
                    ..MatmulOperands::all(None)
                },
                ..problem
            },
            "tiles",
        ),
        None => (problem, "rows"),
    }
}

/// What one routine made of a product: how many of its settings prepare it, out of how many, and
/// the distinct refusals of the others.
struct Served {
    prepared: usize,
    settings: usize,
    refusals: Vec<String>,
}

/// `routine` at every setting of `lists`, each counted once, on `problem`.
fn served<R>(
    routine: R,
    hw: &Hardware,
    problem: &MatmulRoutineProblem,
    lists: impl IntoIterator<Item = Vec<R::Settings>>,
) -> Served
where
    R: Routine<Problem = MatmulRoutineProblem>,
    R::Settings: PartialEq,
{
    let mut settings: Vec<R::Settings> = Vec::new();
    for setting in lists.into_iter().flatten() {
        if !settings.contains(&setting) {
            settings.push(setting);
        }
    }
    let mut prepared = 0;
    let mut refusals = Vec::new();
    for setting in &settings {
        match routine.prepare(hw, problem, setting) {
            Ok(_) => prepared += 1,
            Err(refusal) => {
                let refusal = refusal.to_string();
                if !refusals.contains(&refusal) {
                    refusals.push(refusal);
                }
            }
        }
    }
    refusals.sort();
    Served {
        prepared,
        settings: settings.len(),
        refusals,
    }
}

/// Every routine on `problem`, named by its prefix, at the settings of every list it races.
fn every_routine(hw: &Hardware, problem: &MatmulRoutineProblem) -> Vec<(&'static str, Served)> {
    vec![
        (
            "unit-registers",
            served(
                UnitRegistersMatmulRoutine,
                hw,
                problem,
                [UnitRegistersMatmulRoutine.settings(hw)],
            ),
        ),
        (
            "unit-whole-k",
            served(
                UnitWholeKMatmulRoutine,
                hw,
                problem,
                [UnitWholeKMatmulRoutine.settings(hw)],
            ),
        ),
        (
            "plane-shuffle",
            served(
                PlaneShuffleMatmulRoutine,
                hw,
                problem,
                [
                    PlaneShuffleMatmulRoutine.settings(hw),
                    PlaneShuffleMatmulRoutine.packed_settings(),
                ],
            ),
        ),
        (
            "plane-dequantize",
            served(
                PlaneDequantizeMatmulRoutine,
                hw,
                problem,
                [PlaneDequantizeMatmulRoutine.settings(hw)],
            ),
        ),
        (
            "cube-shuffle",
            served(
                CubeShuffleMatmulRoutine,
                hw,
                problem,
                [
                    CubeShuffleMatmulRoutine.settings(hw),
                    CubeShuffleMatmulRoutine.packed_settings(hw),
                ],
            ),
        ),
        (
            "cube-staged",
            served(
                CubeStagedMatmulRoutine,
                hw,
                problem,
                [
                    CubeStagedMatmulRoutine.settings(hw),
                    CubeStagedMatmulRoutine.packed_settings(hw),
                    CubeStagedMatmulRoutine.packed_one_step_settings(),
                ],
            ),
        ),
        (
            "cube-staged-async-copy",
            served(
                CubeStagedAsyncCopyMatmulRoutine,
                hw,
                problem,
                [
                    CubeStagedAsyncCopyMatmulRoutine.settings(hw),
                    CubeStagedAsyncCopyMatmulRoutine.packed_settings(hw),
                    CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(hw),
                ],
            ),
        ),
        (
            "cube-staged-tma",
            served(
                CubeStagedTmaMatmulRoutine,
                hw,
                problem,
                [
                    CubeStagedTmaMatmulRoutine.settings(hw),
                    CubeStagedTmaMatmulRoutine.block_scaled_settings(hw),
                ],
            ),
        ),
        (
            "cube-staged-tma-filler",
            served(
                CubeStagedTmaFillerMatmulRoutine,
                hw,
                problem,
                [
                    CubeStagedTmaFillerMatmulRoutine.settings(hw),
                    CubeStagedTmaFillerMatmulRoutine.packed_settings(hw),
                ],
            ),
        ),
        (
            "cube-staged-tma-stored",
            served(
                CubeStagedTmaStoredMatmulRoutine,
                hw,
                problem,
                [
                    CubeStagedTmaStoredMatmulRoutine.settings(hw),
                    CubeStagedTmaStoredMatmulRoutine.block_scaled_settings(hw),
                ],
            ),
        ),
        (
            "cube-staged-wgmma",
            served(
                CubeStagedWgmmaMatmulRoutine,
                hw,
                problem,
                [CubeStagedWgmmaMatmulRoutine.settings(hw)],
            ),
        ),
        (
            "hypercube-last-arrival",
            served(
                HypercubeLastArrivalMatmulRoutine,
                hw,
                problem,
                [
                    HypercubeLastArrivalMatmulRoutine.settings(hw),
                    HypercubeLastArrivalMatmulRoutine.packed_settings(hw),
                ],
            ),
        ),
    ]
}

/// Every product on every device: a heading per product, the routines that serve it, then each
/// refusal of a routine that serves it at no setting, once, with the routines that gave it.
fn render_all() -> String {
    let mut out = String::new();
    for (device, hw) in machines() {
        for rows in ROWS {
            for (activation, weight) in pairings() {
                let (problem, layout) = product(&hw, rows, activation, weight);
                out.push_str(&format!(
                    "# {device} · {rows} rows · {} × {} ({layout})\n",
                    activation.name(),
                    weight.name()
                ));
                let routines = every_routine(&hw, &problem);
                let serving: Vec<String> = routines
                    .iter()
                    .filter(|(_, served)| served.prepared > 0)
                    .map(|(name, served)| format!("{name} {}/{}", served.prepared, served.settings))
                    .collect();
                match serving.is_empty() {
                    true => out.push_str("served by: none\n"),
                    false => out.push_str(&format!("served by: {}\n", serving.join(", "))),
                }
                let mut refused: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
                for (name, served) in &routines {
                    if served.prepared == 0 {
                        for refusal in &served.refusals {
                            refused.entry(refusal).or_default().push(name);
                        }
                    }
                }
                for (refusal, names) in refused {
                    out.push_str(&format!("  refused ({}): {refusal}\n", names.join(", ")));
                }
                out.push('\n');
            }
        }
    }
    out
}
