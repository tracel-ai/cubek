use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub struct Footprint {
    /// The discrete source indices to gather.
    pub indices: SmallVec<[i64; 8]>,

    /// The computed weights corresponding to each tap.
    pub weights: SmallVec<[f32; 8]>,
}
