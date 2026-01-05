#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// Number of elements each cube covers in the tensors
pub struct CubeSpan {
    pub m: u32,
    pub n: u32,
    pub batch: u32,
}
