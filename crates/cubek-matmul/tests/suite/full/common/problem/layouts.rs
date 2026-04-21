pub mod rr {
    use super::*;
    use cubek_std::MatrixLayout;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
    }

    include!("problem_size.rs");
}

pub mod rc {
    use super::*;
    use cubek_std::MatrixLayout;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
    }

    include!("problem_size.rs");
}

pub mod cr {
    use super::*;
    use cubek_std::MatrixLayout;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::ColMajor, MatrixLayout::RowMajor)
    }

    include!("problem_size.rs");
}

pub mod cc {
    use super::*;
    use cubek_std::MatrixLayout;

    pub fn layouts() -> (MatrixLayout, MatrixLayout) {
        (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
    }

    include!("problem_size.rs");
}
