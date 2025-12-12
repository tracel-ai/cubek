#[cfg(all(not(feature = "matmul_tests_layouts"),))]
pub mod default {
    use super::*;
    use cubek_matmul::components::MatrixLayout;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
    }

    include!("problem_size.rs");
}

#[cfg(feature = "matmul_tests_layouts")]
mod rr {
    use super::*;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
    }

    include!("problem_size.rs");
}

#[cfg(feature = "matmul_tests_layouts")]
mod rc {
    use super::*;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
    }

    include!("problem_size.rs");
}

#[cfg(feature = "matmul_tests_layouts")]
mod cr {
    use super::*;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::ColMajor, MatrixLayout::RowMajor)
    }

    include!("problem_size.rs");
}

#[cfg(feature = "matmul_tests_layouts")]
mod cc {
    use super::*;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
    }

    include!("problem_size.rs");
}
