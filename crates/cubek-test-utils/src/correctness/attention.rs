//! The attention of a packed cache as the host computes it: the reference its reads are checked
//! against.

/// Vectors as a packed cache stores them, read on the host: vector `v`'s `coords` coordinates lie
/// low first in its words as `bits`-wide indices into `codebook`, and the vector is scaled by
/// `norms[v]`.
pub struct PackedVectors<'a> {
    /// Every vector's words, one vector's after another's.
    pub words: &'a [u32],
    pub norms: &'a [f32],
    pub codebook: &'a [f32],
    pub bits: u32,
    pub coords: usize,
}

impl PackedVectors<'_> {
    /// Coordinate `coord` of vector `vector`, decoded.
    pub fn decoded(&self, vector: usize, coord: usize) -> f64 {
        let per_word = 32 / self.bits as usize;
        let word = self.words[vector * self.coords.div_ceil(per_word) + coord / per_word];
        let index = (word >> ((coord % per_word) as u32 * self.bits)) & ((1 << self.bits) - 1);
        self.codebook[index as usize] as f64 * self.norms[vector] as f64
    }
}

/// One query's attention on the host, in `f64`: the softmax of its dot products with the keys it
/// sees, each times `scale`, weighting their values.
pub struct HostAttention {
    pub scale: f64,
}

impl HostAttention {
    /// The output, `val_dim` wide, of `query` over the keys `seen` names: key `j`'s coordinate `d`
    /// is `key(j, d)`, and its value's coordinate `c` is `value(j, c)`. A query that sees no key
    /// outputs zeros.
    pub fn attend(
        &self,
        query: &[f64],
        seen: &[usize],
        key: impl Fn(usize, usize) -> f64,
        value: impl Fn(usize, usize) -> f64,
        val_dim: usize,
    ) -> Vec<f64> {
        let scores: Vec<f64> = seen
            .iter()
            .map(|&j| {
                let dot: f64 = query.iter().enumerate().map(|(d, q)| q * key(j, d)).sum();
                dot * self.scale
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        (0..val_dim)
            .map(|c| {
                seen.iter()
                    .zip(&weights)
                    .map(|(&j, weight)| weight / total * value(j, c))
                    .sum()
            })
            .collect()
    }
}
