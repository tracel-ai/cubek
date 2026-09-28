//! The transports a quantized operand takes: unpacking a stage narrower than the source's words,
//! and staging the scales beside the words they belong to.

use cubecl::{
    prelude::*,
    std::quant::unpack_fields,
    std::tensor::{AsView, AsViewExpand},
};

use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// The sub-word twin of [`scan_transparent`](Memory::scan_transparent): the source's served
    /// line is one whole packed word, unpacked into `num_quants / W` lines of this store's width,
    /// which is how a packed operand stages on a device whose vectors cannot cover a word.
    ///
    /// Unchecked only and unreachable any other way (a checked operand cannot vectorize), so the
    /// assert below is a backstop for hand-built args; the ragged tail is the caller's ordinary
    /// `checked(false)` claim. The innermost scale block must cover whole words.
    pub(crate) fn scan_words<W: Size>(&mut self, src: &Memory<T>) {
        #[comptime]
        match &src.store.quant {
            ComptimeOption::Some(info) => {
                let nq = comptime!(info.scheme.num_quants());
                comptime!(assert!(
                    src.store.vector_size == nq,
                    "Memory::scan_words: the source serves whole words (vector_size == num_quants)"
                ));
                let w = comptime!(self.store.vector_size);
                comptime!(assert!(
                    w < nq && nq.is_multiple_of(w),
                    "Memory::scan_words: the stage width must divide the packing factor"
                ));
                comptime!(assert!(
                    !src.access.overhang.masks(),
                    "Memory::scan_words: a sub-word fill reads unchecked"
                ));
                comptime!(assert!(
                    info.block.last().unwrap().is_multiple_of(nq),
                    "Memory::scan_words: the innermost scale block must cover whole words"
                ));
                let lpw = comptime!(nq / w);
                let size!(NW) = 1usize;
                let words = src
                    .lines_storage::<u32, NW>()
                    .view(src.base())
                    .view(src.window())
                    .view(FlatLayout::new(src.window.extent.clone()));
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(src.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(FlatLayout::new(src.window.extent.clone()));
                let mut d = self.flat_mut::<W>();
                // A word and its scale are read once, and the lines under them are built: the
                // `lpw` lines of word `word` start at `word · lpw`, each `w` fields further in.
                let total = d.shape() / lpw;
                let workers = CUBE_DIM as usize;
                let mut word = UNIT_POS as usize;
                while word < total {
                    let bits = words.read(word).extract(0usize);
                    let scale = Vector::new(T::cast_from(scales.read(word)));
                    #[unroll]
                    for l in 0..lpw {
                        let vals = unpack_fields::<T, W>(
                            bits,
                            comptime!((l * w) as u32),
                            info.table.clone(),
                            comptime!(info.scheme),
                        );
                        d.write(word * lpw + l, vals * scale);
                    }
                    word += workers;
                }
            }
            ComptimeOption::None => {
                panic!("Memory::scan_words: a plain source has no words to unpack")
            }
        }
    }

    /// Refill quantized stage's scales side-channel from `src`, cooperatively across the cube:
    /// one f32 per block of the sub-tile into the row-major grid [`smem_quant_info`] laid out, each
    /// read by dotting its block coords with `src`'s scale strides from `src`'s `window_start`.
    pub(crate) fn stage_scales(&mut self, src: &Memory<T>) {
        let dst = self.store.quant.as_mut().unwrap();
        let sinfo = src.store.quant.as_ref().unwrap();
        let nb = comptime!(dst.scale_shape.clone());
        let rank = comptime!(nb.len());
        let count = comptime!(nb.iter().product::<usize>());
        let dend = dst.buffer.len();
        let dst_scales = dst.buffer.slice_mut(0, dend);
        let send = sinfo.buffer.len();
        let src_scales = sinfo.buffer.slice(0, send);
        let workers = CUBE_DIM as usize;
        let mut bl = UNIT_POS as usize;
        while bl < count {
            let x = bl.cast::<u32>();
            let mut src_idx = sinfo.window_start;
            #[unroll]
            for p in 0..rank {
                let after = comptime!(nb[(p + 1)..].iter().product::<usize>());
                let bi = x
                    .divided_by(comptime!(after as u32))
                    .remainder(comptime!(nb[p] as u32));
                src_idx = src_idx.plus(bi.times(sinfo.strides.at(p)));
            }
            // The grid holds *effective* scales: a two-level source's global level folds in here,
            // once per block per stage, so everything below the stage serves a one-level scheme
            // and no global scale threads past this point.
            dst_scales[bl] = sinfo.known.effective(src_scales[src_idx.cast::<usize>()]);
            bl += workers;
        }
    }
}
