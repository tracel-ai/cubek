use cubecl::zspace::{Shape, Strides};

#[derive(Debug, PartialEq, Eq, Default)]
pub enum StrideSpec {
    #[default]
    RowMajor,
    ColMajor,
    Custom(Vec<usize>),
}

/// Number of elements in the physical buffer required to cover every logical
/// index in `shape` under `strides`, assuming element 0 is at offset 0.
///
/// Exceeds `shape.iter().product()` for jumpy strides (e.g. a slice stepping
/// over padding) and is less than it for broadcast strides (a stride of 0
/// makes every index in that dim share the same physical offset).
pub fn physical_extent(shape: &Shape, strides: &Strides) -> usize {
    let mut max_offset = 0usize;
    for (s, d) in strides.iter().zip(shape.iter()) {
        if *d > 0 && *s > 0 {
            max_offset += (d - 1) * s;
        }
    }
    max_offset + 1
}

impl StrideSpec {
    pub fn compute_strides(&self, shape: &Shape) -> Strides {
        let n = shape.len();
        match self {
            StrideSpec::RowMajor => {
                assert!(n >= 2, "RowMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 1] = 1;
                for i in (0..n - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                Strides::new(&strides)
            }
            StrideSpec::ColMajor => {
                assert!(n >= 2, "ColMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 2] = 1;
                strides[n - 1] = shape[n - 2];
                for i in (0..n - 2).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                Strides::new(&strides)
            }
            StrideSpec::Custom(strides) => {
                assert!(
                    strides.len() == n,
                    "Custom strides must have the same rank as the shape"
                );
                strides.clone().into()
            }
        }
    }
}
