use cubecl::{self as cubecl, cube, prelude::*, std::FastDivmod};

/// A unit for every block each sequence lists, writing the block's page in a cache read as pages of
/// one block each: sequence `b`'s `j`-th listed block, `blocks[b, j]`, lies in its page `pages[b,
/// blocks[b, j] / blocks_per_page]`, at that page's block `blocks[b, j] % blocks_per_page`, and the
/// view steps `page_blocks` blocks from one page to the next. The unit at absolute position `b ·
/// listed + j`, `listed` the blocks each sequence lists, writes `block_pages[b, j]`.
///
/// A block past a sequence's pages lies in its last page, and a page past the pool, `last_page`
/// its last, is that last page: a sequence reads no other sequence's pages, and no block past the
/// view.
///
/// - `blocks`: the blocks each sequence lists, `u32` `[b, listed]`.
/// - `pages`: the pages of the cache each sequence holds, `u32` `[b, pages_per_sequence]`.
/// - `block_pages`: each listed block's page of the view, `u32` `[b, listed]`, written.
/// - `listed`, `blocks_per_page`, `page_blocks`, `last_page`: as above.
#[cube(launch)]
pub fn block_pages_kernel(
    blocks: &Tensor<u32>,
    pages: &Tensor<u32>,
    block_pages: &mut Tensor<u32>,
    listed: FastDivmod<u32>,
    blocks_per_page: FastDivmod<u32>,
    page_blocks: u32,
    last_page: u32,
) {
    if ABSOLUTE_POS >= block_pages.len() {
        terminate!();
    }
    let (sequence, slot) = listed.div_mod(ABSOLUTE_POS as u32);
    let (sequence, slot) = (sequence as usize, slot as usize);
    let block = blocks[sequence * blocks.stride(0) + slot * blocks.stride(1)];
    let (column, within) = blocks_per_page.div_mod(block);
    let column = (column as usize).min(pages.shape(1) - 1);
    let page = pages[sequence * pages.stride(0) + column * pages.stride(1)].min(last_page);
    let at = sequence * block_pages.stride(0) + slot * block_pages.stride(1);
    block_pages[at] = page * page_blocks + within;
}
