use crate::attention::packed_cache::packed_fields;
use crate::attention::packed_cache::{KeyFixup, PackedCacheArg};
use crate::attention::{B, KV, S, StepLines, StepLinesExpand, StepWalk};
use cubecl::tile::Region;
use cubecl::{self as cubecl, cube, prelude::*};

/// The keys or the values of a packed cache, read as lines ([`PackedCacheArg`]).
///
/// With a fix-up, the operand is the keys, whose lines run twice the head dim: the first half the
/// scaled codebook entries, the second the same indices read through `sign_codebook`, scaled by
/// the vector's entry of `errs`. A query whose second half is its projection then scores both
/// terms in one dot.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct PackedLines<'a> {
    pub words: &'a Tensor<u32>,
    pub norms: &'a Tensor<f32>,
    pub codebook: &'a Tensor<f32>,
    /// Without a fix-up, any tensor: these are never read.
    pub errs: &'a Tensor<f32>,
    pub sign_codebook: &'a Tensor<f32>,
    /// Coordinates a vector holds: the head dim.
    #[cube(comptime)]
    pub coords: usize,
    #[cube(comptime)]
    pub bits: u32,
    #[cube(comptime)]
    pub fixup: KeyFixup,
}

#[cube]
impl<'a> PackedLines<'a> {
    /// The keys of `cache`, their fix-up read where they carry one.
    pub fn keys(cache: &'a PackedCacheArg) -> PackedLines<'a> {
        PackedLines::<'a> {
            words: &cache.key_words,
            norms: &cache.key_norms,
            codebook: &cache.key_codebook,
            errs: &cache.key_errs,
            sign_codebook: &cache.key_sign_codebook,
            coords: comptime!(cache.coords),
            bits: comptime!(cache.key_bits),
            fixup: comptime!(cache.fixup),
        }
    }

    /// The values of `cache`, which carry no fix-up.
    pub fn values(cache: &'a PackedCacheArg) -> PackedLines<'a> {
        PackedLines::<'a> {
            words: &cache.value_words,
            norms: &cache.value_norms,
            codebook: &cache.value_codebook,
            errs: &cache.value_norms,
            sign_codebook: &cache.value_codebook,
            coords: comptime!(cache.coords),
            bits: comptime!(cache.value_bits),
            fixup: KeyFixup::RoundOnly,
        }
    }
}

#[cube]
impl<'a, N: Size> StepLines<N> for PackedLines<'a> {
    type Element = f32;

    fn load(
        &self,
        step: &Region,
        walk: &StepWalk,
        #[comptime] lines: usize,
    ) -> Array<Vector<f32, N>> {
        let width = N::value();
        let coords = comptime!(self.coords);
        let (batch, head, first_key) = (step.origin(B), step.origin(KV), step.origin(S));
        let (teams, keys_in_flight) = comptime!((walk.teams, walk.keys_in_flight));
        let units_per_key = comptime!(walk.units_per_key);
        let per_key = comptime!(lines.div_ceil(units_per_key));
        let mut loaded = Array::<Vector<f32, N>>::new(comptime!(keys_in_flight * per_key));
        #[unroll]
        for c in 0..keys_in_flight {
            let s = first_key + c * teams + walk.team;
            let seen = (s as u32) < walk.attended;
            let row = batch * self.words.stride(0)
                + head * self.words.stride(1)
                + s * self.words.stride(2);
            let norm_at = batch * self.norms.stride(0)
                + head * self.norms.stride(1)
                + s * self.norms.stride(2);
            #[unroll]
            for p in 0..per_key {
                let line = walk.lane + p * units_per_key;
                let mut value = Vector::<f32, N>::cast_from(0u32);
                if line < lines && seen {
                    let first = line * width;
                    if first < coords {
                        let fields = packed_fields::<f32, N>(
                            self.words,
                            row,
                            first,
                            self.codebook,
                            comptime!(self.bits),
                        );
                        value = fields * Vector::cast_from(self.norms[norm_at]);
                    } else if comptime!(self.fixup == KeyFixup::Qjl) {
                        let fields = packed_fields::<f32, N>(
                            self.words,
                            row,
                            first - coords,
                            self.sign_codebook,
                            comptime!(self.bits),
                        );
                        let err_at = batch * self.errs.stride(0)
                            + head * self.errs.stride(1)
                            + s * self.errs.stride(2);
                        value = fields * Vector::cast_from(self.errs[err_at]);
                    }
                }
                loaded[c * per_key + p] = value;
            }
        }
        loaded
    }

    /// The same lines: a packed cache is read by each key's position, through no window.
    fn at(&self, _region: &Region) -> PackedLines<'a> {
        self.clone()
    }
}
