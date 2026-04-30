use cubek::std::MatrixLayout;

use crate::registry::ItemDescriptor;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

pub struct GemmProblem {
    pub b: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub precision: Precision,
}

#[derive(Clone, Copy)]
struct ShapeSpec {
    tag: &'static str,
    label: &'static str,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
}

/// Historical shapes from the bench harness. Batch counts come from the
/// previous `entry()` heuristic (target ~2 * 6144^3 ops, b clamped to a power
/// of two and to 4096) or from the explicit tuples that bypassed it.
const SHAPES: &[ShapeSpec] = &[
    // Default vector × matrix.
    ShapeSpec {
        tag: "vecmat_2x1x4096x4096",
        label: "VecMat (b=2 m=1 n=4096 k=4096)",
        b: 2,
        m: 1,
        n: 4096,
        k: 4096,
    },
    // Square.
    ShapeSpec {
        tag: "square_1x8192",
        label: "Square (b=1 8192³)",
        b: 1,
        m: 8192,
        n: 8192,
        k: 8192,
    },
    ShapeSpec {
        tag: "square_1x6144",
        label: "Square (b=1 6144³)",
        b: 1,
        m: 6144,
        n: 6144,
        k: 6144,
    },
    ShapeSpec {
        tag: "square_2x4096",
        label: "Square (b=2 4096³)",
        b: 2,
        m: 4096,
        n: 4096,
        k: 4096,
    },
    ShapeSpec {
        tag: "square_16x2048",
        label: "Square (b=16 2048³)",
        b: 16,
        m: 2048,
        n: 2048,
        k: 2048,
    },
    ShapeSpec {
        tag: "square_2x1024",
        label: "Square (b=2 1024³)",
        b: 2,
        m: 1024,
        n: 1024,
        k: 1024,
    },
    ShapeSpec {
        tag: "square_1024x512",
        label: "Square (b=1024 512³)",
        b: 1024,
        m: 512,
        n: 512,
        k: 512,
    },
    // Skinny: large n, small m and k.
    ShapeSpec {
        tag: "skinny_64_1024_64",
        label: "Skinny (b=4096 m=64 n=1024 k=64)",
        b: 4096,
        m: 64,
        n: 1024,
        k: 64,
    },
    ShapeSpec {
        tag: "skinny_32_1024_32",
        label: "Skinny (b=4096 m=32 n=1024 k=32)",
        b: 4096,
        m: 32,
        n: 1024,
        k: 32,
    },
    ShapeSpec {
        tag: "skinny_10_1024_10",
        label: "Skinny (b=4096 m=10 n=1024 k=10)",
        b: 4096,
        m: 10,
        n: 1024,
        k: 10,
    },
    // Skinny: large k, small m and n.
    ShapeSpec {
        tag: "skinny_64_64_1024",
        label: "Skinny (b=4096 m=64 n=64 k=1024)",
        b: 4096,
        m: 64,
        n: 64,
        k: 1024,
    },
    ShapeSpec {
        tag: "skinny_32_32_1024",
        label: "Skinny (b=4096 m=32 n=32 k=1024)",
        b: 4096,
        m: 32,
        n: 32,
        k: 1024,
    },
    ShapeSpec {
        tag: "skinny_10_10_1024",
        label: "Skinny (b=4096 m=10 n=10 k=1024)",
        b: 4096,
        m: 10,
        n: 10,
        k: 1024,
    },
    // Skinny: large m, small n and k.
    ShapeSpec {
        tag: "skinny_1024_64_64",
        label: "Skinny (b=4096 m=1024 n=64 k=64)",
        b: 4096,
        m: 1024,
        n: 64,
        k: 64,
    },
    ShapeSpec {
        tag: "skinny_1024_32_32",
        label: "Skinny (b=4096 m=1024 n=32 k=32)",
        b: 4096,
        m: 1024,
        n: 32,
        k: 32,
    },
    ShapeSpec {
        tag: "skinny_1024_10_10",
        label: "Skinny (b=4096 m=1024 n=10 k=10)",
        b: 4096,
        m: 1024,
        n: 10,
        k: 10,
    },
    // Misc rectangular.
    ShapeSpec {
        tag: "rect_16x1x2048x8192",
        label: "Rect (b=16 m=1 n=2048 k=8192)",
        b: 16,
        m: 1,
        n: 2048,
        k: 8192,
    },
    ShapeSpec {
        tag: "rect_16x1x4096x4096",
        label: "Rect (b=16 m=1 n=4096 k=4096)",
        b: 16,
        m: 1,
        n: 4096,
        k: 4096,
    },
    ShapeSpec {
        tag: "rect_1x512x512x512",
        label: "Rect (b=1 512³)",
        b: 1,
        m: 512,
        n: 512,
        k: 512,
    },
    // Edge cases (degenerate dims).
    ShapeSpec {
        tag: "outer_2x8192x8192x1",
        label: "Outer (b=2 m=8192 n=8192 k=1)",
        b: 2,
        m: 8192,
        n: 8192,
        k: 1,
    },
    ShapeSpec {
        tag: "matvec_2x8192x1x8192",
        label: "MatVec (b=2 m=8192 n=1 k=8192)",
        b: 2,
        m: 8192,
        n: 1,
        k: 8192,
    },
    ShapeSpec {
        tag: "vecmat_2x1x8192x8192",
        label: "VecMat (b=2 m=1 n=8192 k=8192)",
        b: 2,
        m: 1,
        n: 8192,
        k: 8192,
    },
];

#[derive(Clone, Copy)]
struct LayoutSpec {
    suffix: &'static str,
    label: &'static str,
    lhs: MatrixLayout,
    rhs: MatrixLayout,
}

const LAYOUTS: &[LayoutSpec] = &[
    LayoutSpec {
        suffix: "rr",
        label: "row/row",
        lhs: MatrixLayout::RowMajor,
        rhs: MatrixLayout::RowMajor,
    },
    LayoutSpec {
        suffix: "rc",
        label: "row/col",
        lhs: MatrixLayout::RowMajor,
        rhs: MatrixLayout::ColMajor,
    },
    LayoutSpec {
        suffix: "cr",
        label: "col/row",
        lhs: MatrixLayout::ColMajor,
        rhs: MatrixLayout::RowMajor,
    },
    LayoutSpec {
        suffix: "cc",
        label: "col/col",
        lhs: MatrixLayout::ColMajor,
        rhs: MatrixLayout::ColMajor,
    },
];

#[derive(Clone, Copy)]
struct PrecisionSpec {
    suffix: &'static str,
    label: &'static str,
    precision: Precision,
}

const PRECISIONS: &[PrecisionSpec] = &[
    PrecisionSpec {
        suffix: "f32",
        label: "f32",
        precision: Precision::F32,
    },
    PrecisionSpec {
        suffix: "f16",
        label: "f16",
        precision: Precision::F16,
    },
];

fn make_id(shape: &ShapeSpec, layout: &LayoutSpec, prec: &PrecisionSpec) -> String {
    format!("{}_{}_{}", shape.tag, layout.suffix, prec.suffix)
}

fn make_label(shape: &ShapeSpec, layout: &LayoutSpec, prec: &PrecisionSpec) -> String {
    format!("{} {} [{}]", shape.label, layout.label, prec.label)
}

pub fn problems() -> Vec<ItemDescriptor> {
    SHAPES
        .iter()
        .flat_map(|shape| {
            LAYOUTS.iter().flat_map(move |layout| {
                PRECISIONS.iter().map(move |prec| ItemDescriptor {
                    id: make_id(shape, layout, prec),
                    label: make_label(shape, layout, prec),
                })
            })
        })
        .collect()
}

pub(crate) fn problem_for(id: &str) -> Option<GemmProblem> {
    SHAPES.iter().find_map(|shape| {
        LAYOUTS.iter().find_map(|layout| {
            PRECISIONS.iter().find_map(|prec| {
                (make_id(shape, layout, prec) == id).then(|| GemmProblem {
                    b: shape.b,
                    m: shape.m,
                    n: shape.n,
                    k: shape.k,
                    lhs_layout: layout.lhs,
                    rhs_layout: layout.rhs,
                    precision: prec.precision,
                })
            })
        })
    })
}
