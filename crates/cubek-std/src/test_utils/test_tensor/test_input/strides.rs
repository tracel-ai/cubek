#[derive(Debug, PartialEq, Eq, Default)]
pub enum StrideSpec {
    #[default]
    RowMajor,
    ColMajor,
    Custom(Vec<usize>),
}

impl StrideSpec {
    pub fn compute_strides(&self, shape: &[usize]) -> Vec<usize> {
        let n = shape.len();

        match self {
            StrideSpec::RowMajor => {
                assert!(n >= 2, "RowMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 1] = 1;
                for i in (0..n - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                strides
            }
            StrideSpec::ColMajor => {
                assert!(n >= 2, "ColMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 2] = 1;
                strides[n - 1] = shape[n - 2];
                for i in (0..n - 2).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                strides
            }
            StrideSpec::Custom(strides) => {
                assert!(
                    strides.len() == n,
                    "Custom strides must have the same rank as the shape"
                );
                strides.clone()
            }
        }
    }
}
