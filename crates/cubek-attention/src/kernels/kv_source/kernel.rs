use cubecl::{self as cubecl, cube, prelude::*, std::FastDivmod};

/// A unit for every block each user lists, writing the block's page in a cache read as pages of
/// one block each: user `b`'s `j`-th listed block, `blocks[b, j]`, lies in its page
/// `pages[b, blocks[b, j] / blocks_per_page]`, at that page's block `blocks[b, j] % blocks_per_page`,
/// and the view steps `page_blocks` blocks from one page to the next. The unit at absolute position
/// `b · listed + j`, `listed` the blocks each user lists, writes `block_pages[b, j]`.
#[cube(launch)]
pub fn block_pages_kernel(
    blocks: &Tensor<u32>,
    pages: &Tensor<u32>,
    block_pages: &mut Tensor<u32>,
    listed: FastDivmod<u32>,
    blocks_per_page: FastDivmod<u32>,
    page_blocks: u32,
) {
    if ABSOLUTE_POS >= block_pages.len() {
        terminate!();
    }
    let (user, slot) = listed.div_mod(ABSOLUTE_POS as u32);
    let (user, slot) = (user as usize, slot as usize);
    let block = blocks[user * blocks.stride(0) + slot * blocks.stride(1)];
    let (column, within) = blocks_per_page.div_mod(block);
    let page = pages[user * pages.stride(0) + column as usize * pages.stride(1)];
    let at = user * block_pages.stride(0) + slot * block_pages.stride(1);
    block_pages[at] = page * page_blocks + within;
}
