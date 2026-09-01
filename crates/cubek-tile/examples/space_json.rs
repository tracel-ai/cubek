//! Prototype: derive a viewer-ready description of a tile [`Space`] from the
//! `Tiling` alone, with nothing hand-written per kernel.
//!
//! The whole derivation is [`emit`]: descend with [`Space::divide`] until
//! [`Space::is_final`], and at each level read the per-axis extent, sub-tile
//! edge, tile count, distribution, walk order and buffering off the public
//! accessors. That is the same descent `geometry.rs` already runs to count
//! instances, so nothing here can drift from what the engine does.
//!
//! Run: `cargo run -p cubek-tile --example space_json`

use cubecl::ir::{ElemType, FloatKind};
use cubek_tile::{
    Axis, Buffering, ComputeScope, Coverage, CubeAxis, Cut, Distribution, Instruction, Operand,
    OperandSet, Partitioner, Residence, Space, Spread, Tiling, WalkOrder,
};

/// The three operands every one of these kernels contracts, threaded through the
/// `Tiling::over` build so each level states where it puts them. Their residence
/// column is the memory hierarchy — it is stated nowhere else.
struct Operands {
    lhs: Operand,
    rhs: Operand,
    out: Operand,
}

impl Operands {
    fn new() -> Self {
        let f16 = ElemType::Float(FloatKind::F16);
        let f32 = ElemType::Float(FloatKind::F32);
        Operands {
            lhs: Operand::new(&[M, K], f16),
            rhs: Operand::new(&[K, N], f16),
            out: Operand::new(&[M, N], f32),
        }
    }
}

impl OperandSet for Operands {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.lhs, &mut self.rhs, &mut self.out].into_iter()
    }
}

fn residence_name(residence: Residence) -> &'static str {
    match residence {
        Residence::InPlace => "InPlace",
        Residence::Smem => "Smem",
        Residence::Register => "Register",
    }
}

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The names an axis carries in the caller's vocabulary; `Axis` is a bare `u8`,
/// so this table is the one thing the emitter cannot derive.
fn name(axis: Axis) -> &'static str {
    match axis {
        M => "M",
        N => "N",
        K => "K",
        Axis(i) => Box::leak(format!("A{i}").into_boxed_str()),
    }
}

// ---------------------------------------------------------------- the emitter

/// Every level of `space`, coarse to fine, plus the leaf tile it bottoms out in.
fn emit(label: &str, space: &Space, ops: &Operands) -> String {
    let mut levels = Vec::new();
    let mut level = space.clone();
    let mut depth = 0;

    while !level.is_final() {
        levels.push(emit_level(&level, depth));
        level = level.divide();
        depth += 1;
    }

    // `level` is now the leaf: the finest tile, no partitioner left to cut it.
    let leaf = format!(
        "{{\"extents\":[{}],\"instruction\":{}}}",
        join(level.axes().map(|a| format!(
            "{{\"axis\":\"{}\",\"extent\":{}}}",
            name(a),
            level.extent(a)
        ))),
        quote_opt(space.instruction().map(instruction_name))
    );

    format!(
        "{{\"label\":\"{label}\",\"axes\":[{}],\"levels\":[{}],\"leaf\":{leaf},\"faces\":[{}]}}",
        join(space.axes().map(|a| format!(
            "{{\"axis\":\"{}\",\"extent\":{},\"dynamic\":{}}}",
            name(a),
            space.extent(a),
            space.is_dynamic(a)
        ))),
        join(levels.into_iter()),
        join(faces(space, ops).into_iter()),
    )
}

/// One level: what it cuts each axis into, and how it hands the tiles out.
fn emit_level(level: &Space, depth: usize) -> String {
    let partitioner = level.partitioner();
    let axes = level.axes().map(|axis| {
        let edge = partitioner.edge(axis);
        let extent = level.extent(axis);
        format!(
            "{{\"axis\":\"{}\",\"extent\":{extent},\"edge\":{edge},\"count\":{},\
              \"overhang\":{},{}}}",
            name(axis),
            level.count(axis),
            !extent.is_multiple_of(edge),
            emit_dist(partitioner.distribution(axis), level.count(axis)),
        )
    });

    format!(
        "{{\"depth\":{depth},\"order\":\"{:?}\",\"buffering\":{},\"scope\":\"{}\",\"axes\":[{}]}}",
        partitioner.order(),
        buffering(partitioner).depth(),
        finest_scope(level),
        join(axes),
    )
}

/// How one axis's tiles are dealt out: the box's color and its badge.
fn emit_dist(dist: Distribution, grid: usize) -> String {
    match dist {
        Distribution::Sequential => {
            "\"scope\":\"Sequential\",\"spread\":null,\"instances\":1".to_string()
        }
        Distribution::Spatial {
            scope,
            spread,
            coverage,
        } => format!(
            "\"scope\":\"{}\",\"spread\":\"{}\",\"instances\":{}",
            scope_name(scope),
            match spread {
                Spread::Contiguous => "Contiguous",
                Spread::Interleaved => "Interleaved",
            },
            // A deferred lane count is stamped at launch; say so rather than guess.
            match coverage {
                Coverage::Instances(n) => n.to_string(),
                Coverage::TilesEach(t) => (grid / t).to_string(),
                Coverage::PlaneLanes => "null".to_string(),
            },
        ),
    }
}

/// The three operand shadows of an operation space: each names the axes it
/// spans, and by omission the ones it is invariant along.
fn faces(space: &Space, ops: &Operands) -> Vec<String> {
    let spans = |axes: &[Axis]| axes.iter().all(|a| space.contains(*a));
    [
        ("lhs", vec![M, K], &ops.lhs, false),
        ("rhs", vec![K, N], &ops.rhs, false),
        ("out", vec![M, N], &ops.out, true),
    ]
    .into_iter()
    .filter(|(_, axes, _, _)| spans(axes))
    .map(|(role, axes, operand, accumulator)| {
        let projected = space.project(&axes);
        // One stage per level, coarse to fine: where this operand's cells sit while
        // the level below runs, and the type they hold there. A level that stated
        // nothing padded to `InPlace`, which is read where it already lies.
        let stages = operand.stages().iter().map(|stage| {
            format!(
                "{{\"residence\":\"{}\",\"dtype\":\"{:?}\"}}",
                residence_name(stage.residence),
                stage.dtype
            )
        });
        // An operand that states no register stage in a space that contracts does not
        // stay where it lies: the instruction answers for its register form. That is
        // how an accumulator lives in a fragment while every level reads `InPlace`.
        let promoted = space.instruction().is_some()
            && !operand
                .residences()
                .iter()
                .any(|r| matches!(r, Residence::Register));
        format!(
            "{{\"role\":\"{role}\",\"axes\":[{}],\"omits\":[{}],\"dtype\":\"{:?}\",\
             \"accumulator\":{accumulator},\"promoted\":{promoted},\"stages\":[{}]}}",
            join(axes.iter().map(|a| format!("\"{}\"", name(*a)))),
            join(
                space
                    .contracting(&projected)
                    .iter()
                    .map(|a| format!("\"{}\"", name(*a)))
            ),
            operand.dtype(),
            join(stages),
        )
    })
    .collect()
}

fn scope_name(scope: ComputeScope) -> &'static str {
    match scope {
        ComputeScope::Cube(CubeAxis::X) => "CubeX",
        ComputeScope::Cube(CubeAxis::Y) => "CubeY",
        ComputeScope::Cube(CubeAxis::Z) => "CubeZ",
        ComputeScope::Plane => "Plane",
        ComputeScope::Unit => "Unit",
    }
}

/// The level's own scope is the finest any of its axes rides.
fn finest_scope(level: &Space) -> &'static str {
    let rank = |s: &str| match s {
        "Unit" => 4,
        "Plane" => 3,
        s if s.starts_with("Cube") => 2,
        _ => 1,
    };
    level
        .axes()
        .map(|a| match level.partitioner().distribution(a) {
            Distribution::Sequential => "Sequential",
            Distribution::Spatial { scope, .. } => scope_name(scope),
        })
        .max_by_key(|s| rank(s))
        .unwrap_or("Sequential")
}

fn buffering(partitioner: &Partitioner) -> Buffering {
    match partitioner {
        Partitioner::Level(level) => level.buffering(),
        Partitioner::Final => Buffering::SINGLE,
    }
}

fn instruction_name(instruction: Instruction) -> String {
    match instruction {
        Instruction::Registers { .. } => "Registers".to_string(),
        Instruction::Cmma => "Cmma".to_string(),
        Instruction::Mma { .. } => "Mma".to_string(),
    }
}

fn join(parts: impl Iterator<Item = String>) -> String {
    parts.collect::<Vec<_>>().join(",")
}

fn quote_opt(value: Option<String>) -> String {
    value.map_or("null".to_string(), |v| format!("\"{v}\""))
}

// ------------------------------------------------- the spaces, as metabolic builds them

const MEMORY_INSTRUCTION: Instruction = Instruction::registers(1);

/// `edge`-wide tiles across a plane's lanes with an explicit count, the way
/// metabolic's gemv states them.
fn unit(edge: usize, spread: Spread, lanes: usize) -> Cut {
    Cut::new(
        edge,
        Distribution::Spatial {
            scope: ComputeScope::Unit,
            spread,
            coverage: Coverage::Instances(lanes),
        },
    )
}

/// metabolic's cmma gemm: cube grid, plane partition, contraction step, fragment
/// grid — with the residences it states, which are the memory hierarchy.
fn gemm_cmma(ops: &mut Operands) -> Space {
    let (m, n, k) = (2048, 2048, 2048);
    let (im, in_, ik) = (16, 16, 16);
    let (cm, cn) = (2, 2);
    let (stage_m, stage_n, stage_k) = (64, 64, 32);

    Tiling::over(ops, &[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l, o| {
            l.axis(M, Cut::cube(CubeAxis::X, stage_m))
                .axis(N, Cut::cube(CubeAxis::Y, stage_n))
                .axis(K, Cut::sequential(stage_k));
            o.lhs.stage(Residence::Smem);
            o.rhs.stage(Residence::Smem);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::plane(cm * im))
                .axis(N, Cut::plane(cn * in_))
                .axis(K, Cut::sequential(stage_k));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(cm * im))
                .axis(N, Cut::sequential(cn * in_))
                .axis(K, Cut::sequential(ik));
            o.lhs.stage(Residence::Register);
            o.rhs.stage(Residence::Register);
        })
        .instruction(Instruction::Cmma, |l, _| {
            l.axis(M, Cut::sequential(im))
                .axis(N, Cut::sequential(in_))
                .axis(K, Cut::sequential(ik));
        })
        .build()
}

/// metabolic's float gemv, row-major K-split. It stages nothing: the weight is read
/// exactly once, so every level leaves both operands where they already lie.
fn gemv_row_k_split(ops: &mut Operands) -> Space {
    let (d_out, d_in) = (11008, 4096);
    let (tile_n, line, lanes) = (128, 4, 32);

    Tiling::over(ops, &[(M, 1), (N, d_out), (K, d_in)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::cube(CubeAxis::X, tile_n))
                .axis(K, Cut::sequential(d_in));
        })
        .instruction(MEMORY_INSTRUCTION, |l, _| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::plane(line))
                .axis(K, unit(d_in / lanes, Spread::Contiguous, lanes));
        })
        .build()
}

/// metabolic's col plane-fold: the shipped quantized-gemv geometry, with two `Unit`
/// cuts side by side whose instance product is the plane width.
fn gemv_col_plane_fold(ops: &mut Operands) -> Space {
    let (d_out, d_in, n) = (11008, 4096, 1);
    let (rows_per_cube, rows_per_plane, rows_per_lane) = (16, 4, 1);
    let (edge, group_lanes) = (8, 8);

    Tiling::over(ops, &[(M, d_out), (N, n), (K, d_in)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::cube(CubeAxis::X, rows_per_cube))
                .axis(N, Cut::sequential(n))
                .axis(K, Cut::sequential(d_in));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::plane(rows_per_plane))
                .axis(N, Cut::sequential(n))
                .axis(K, Cut::sequential(d_in));
        })
        .instruction(MEMORY_INSTRUCTION, |l, _| {
            l.axis(
                M,
                unit(
                    rows_per_lane,
                    Spread::Contiguous,
                    rows_per_plane / rows_per_lane,
                ),
            )
            .axis(N, Cut::sequential(n))
            .axis(K, unit(edge, Spread::Interleaved, group_lanes));
        })
        .build()
}

fn main() {
    let built: Vec<(&str, Space, Operands)> = vec![
        {
            let mut ops = Operands::new();
            let space = gemm_cmma(&mut ops);
            ("gemm cmma 2048^3", space, ops)
        },
        {
            let mut ops = Operands::new();
            let space = gemv_row_k_split(&mut ops);
            ("gemv float row k-split", space, ops)
        },
        {
            let mut ops = Operands::new();
            let space = gemv_col_plane_fold(&mut ops);
            ("gemv col plane-fold", space, ops)
        },
    ];
    println!(
        "[{}]",
        join(
            built
                .iter()
                .map(|(label, space, ops)| emit(label, space, ops))
        )
    );
}
