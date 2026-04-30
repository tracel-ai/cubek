use cubek::quantization::scheme::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

use crate::registry::ItemDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

impl Layout {
    pub fn short(self) -> &'static str {
        match self {
            Layout::RowMajor => "r",
            Layout::ColMajor => "c",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantSide {
    LhsOnly,
    RhsOnly,
    Both,
}

impl QuantSide {
    fn short(self) -> &'static str {
        match self {
            QuantSide::LhsOnly => "lhs",
            QuantSide::RhsOnly => "rhs",
            QuantSide::Both => "both",
        }
    }
}

#[derive(Clone, Copy)]
pub enum Mode {
    Float,
    Quant {
        scheme: QuantScheme,
        side: QuantSide,
    },
}

#[derive(Clone)]
pub struct QuantizedMatmulProblem {
    pub b: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: Layout,
    pub rhs_layout: Layout,
    pub mode: Mode,
    pub mode_label: String,
}

fn scheme_tensor(value: QuantValue) -> QuantScheme {
    QuantScheme::default()
        .with_mode(QuantMode::Symmetric)
        .with_level(QuantLevel::Tensor)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
}

fn scheme_block(value: QuantValue, block: u8) -> QuantScheme {
    scheme_tensor(value).with_level(QuantLevel::block([block]))
}

fn quant_schemes() -> Vec<(&'static str, QuantScheme)> {
    vec![
        ("q8s-tensor", scheme_tensor(QuantValue::Q8S)),
        ("q4s-tensor", scheme_tensor(QuantValue::Q4S)),
        ("q8s-block32", scheme_block(QuantValue::Q8S, 32)),
        ("q4s-block32", scheme_block(QuantValue::Q4S, 32)),
    ]
}

fn quant_sides() -> Vec<QuantSide> {
    vec![QuantSide::LhsOnly, QuantSide::RhsOnly, QuantSide::Both]
}

fn layouts() -> Vec<(Layout, Layout)> {
    use Layout::*;
    vec![
        (RowMajor, RowMajor),
        (RowMajor, ColMajor),
        (ColMajor, RowMajor),
        (ColMajor, ColMajor),
    ]
}

fn gemm_shapes() -> Vec<(usize, usize, usize, usize)> {
    vec![
        (1, 1024, 1024, 1024),
        (1, 4096, 4096, 4096),
        (2, 1024, 1024, 1024),
    ]
}

fn gemv_shapes() -> Vec<(usize, usize, usize, usize)> {
    vec![(1, 4096, 1, 4096)]
}

fn shape_kind(b: usize, m: usize, n: usize, k: usize) -> &'static str {
    if m == 1 || n == 1 {
        "gemv"
    } else {
        let _ = (b, k);
        "gemm"
    }
}

fn build_id(
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: Layout,
    rhs: Layout,
    mode_label: &str,
) -> String {
    format!(
        "{}_b{}_m{}_n{}_k{}_{}{}_{}",
        shape_kind(b, m, n, k),
        b,
        m,
        n,
        k,
        lhs.short(),
        rhs.short(),
        mode_label,
    )
}

fn build_label(
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: Layout,
    rhs: Layout,
    mode_label: &str,
) -> String {
    format!(
        "{} (b={} m={} n={} k={}) {}{} {}",
        shape_kind(b, m, n, k).to_uppercase(),
        b,
        m,
        n,
        k,
        lhs.short(),
        rhs.short(),
        mode_label,
    )
}

fn modes() -> Vec<(String, Mode)> {
    let mut out = vec![("float".to_string(), Mode::Float)];
    for (name, scheme) in quant_schemes() {
        for side in quant_sides() {
            out.push((
                format!("{}-{}", name, side.short()),
                Mode::Quant { scheme, side },
            ));
        }
    }
    out
}

fn all_specs() -> Vec<QuantizedMatmulProblem> {
    let mut out = Vec::new();
    for shapes in [gemm_shapes(), gemv_shapes()] {
        for (b, m, n, k) in shapes {
            for (lhs_layout, rhs_layout) in layouts() {
                for (mode_label, mode) in modes() {
                    out.push(QuantizedMatmulProblem {
                        b,
                        m,
                        n,
                        k,
                        lhs_layout,
                        rhs_layout,
                        mode,
                        mode_label,
                    });
                }
            }
        }
    }
    out
}

pub fn problems() -> Vec<ItemDescriptor> {
    all_specs()
        .into_iter()
        .map(|p| {
            let id = build_id(
                p.b,
                p.m,
                p.n,
                p.k,
                p.lhs_layout,
                p.rhs_layout,
                &p.mode_label,
            );
            let label = build_label(
                p.b,
                p.m,
                p.n,
                p.k,
                p.lhs_layout,
                p.rhs_layout,
                &p.mode_label,
            );
            ItemDescriptor { id, label }
        })
        .collect()
}

pub(crate) fn problem_for(id: &str) -> Option<QuantizedMatmulProblem> {
    all_specs().into_iter().find(|p| {
        build_id(
            p.b,
            p.m,
            p.n,
            p.k,
            p.lhs_layout,
            p.rhs_layout,
            &p.mode_label,
        ) == id
    })
}
