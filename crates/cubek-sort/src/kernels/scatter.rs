use super::warp_utils::{compute_peer_mask, count_lower_peers, count_set_bits, find_first_set_bit};
use crate::components::config::{NUM_BUCKETS, RADIX_BITS};
use crate::components::key::{Radix, SortKey};
use cubecl::prelude::*;
use cubecl_std::tensor::layout::linear::LinearView;

const NUM_BUCKETS_U32: u32 = NUM_BUCKETS as u32;

#[cube(launch_unchecked)]
pub fn scatter_kernel<KIn: SortKey<Radix = R>, KOut: SortKey<Radix = R>, R: Radix>(
    keys_in: &LinearView<KIn>,
    keys_out: &mut Tensor<KOut>,
    values_in: &LinearView<u32>,
    values_out: &mut Tensor<u32>,
    block_offsets: &Tensor<u32>,
    num_items: u32,
    pass: u32,
    reverse_output_flag: u32,
    #[comptime] items_per_thread: u32,
    #[comptime] has_values: bool,
    #[comptime] num_planes: u32,
    #[comptime] items_per_block: u32,
) {
    let reverse_output = reverse_output_flag != 0;
    // Plane histograms for warp-level ranking (atomic for concurrent updates)
    let plane_hists = SharedMemory::<Atomic<u32>>::new((num_planes as usize) * NUM_BUCKETS);
    // Shared memory buffer for keys (reused for local reordering) - uses Radix type
    let mut shared_keys = SharedMemory::<R>::new(items_per_block as usize);
    // Shared memory buffer for values - only allocate if needed
    let mut shared_values = SharedMemory::<u32>::new(if has_values {
        items_per_block as usize
    } else {
        1 // Minimum allocation when unused
    });
    // Where each digit starts in shared memory (exclusive prefix sum of digit counts)
    let mut digit_start = SharedMemory::<u32>::new(NUM_BUCKETS);
    // Global write offset for each digit (block_offset - digit_start, so global_pos = this + local_idx)
    let mut digit_global = SharedMemory::<u32>::new(NUM_BUCKETS);

    let block_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lane_id = UNIT_POS_PLANE;
    let plane_id = PLANE_POS;

    let block_start = block_id * items_per_block;
    let sub_part_size = PLANE_DIM * items_per_thread;
    let sub_part_start = block_start + plane_id * sub_part_size;

    // Initialize plane histograms
    // With 512 threads, 16 planes * 256 buckets = 4096 entries = 8 per thread
    let total_hist_entries = (num_planes as usize) * NUM_BUCKETS;
    #[allow(clippy::manual_div_ceil)]
    let entries_per_thread = (total_hist_entries + CUBE_DIM as usize - 1) / CUBE_DIM as usize;
    for i in 0..entries_per_thread {
        let idx = thread_id as usize + i * CUBE_DIM as usize;
        if idx < total_hist_entries {
            plane_hists[idx].store(0u32);
        }
    }
    sync_cube();

    // Register arrays for per-thread data - uses Radix type for keys
    // Note: digits recomputed from keys in Phase 3 to reduce register pressure
    let mut keys = Array::<R>::new(items_per_thread as usize);
    let mut values = Array::<u32>::new(items_per_thread as usize);
    let mut local_offsets = Array::<u32>::new(items_per_thread as usize);

    // Compute shift amount and digit mask using cast
    let shift_u32 = pass * RADIX_BITS as u32;
    let shift = R::cast_from(shift_u32);
    let digit_mask = R::cast_from(0xFFu32);

    // Check if this entire plane is full (all items valid) - enables bounds check elimination
    let is_full_plane = (sub_part_start + sub_part_size) <= num_items;

    // Use max radix value for invalid positions (sorts to end in ascending order)
    let max_radix = R::max_value();

    // Phase 1: Load keys and compute warp-level ranking
    // When is_full_plane is true, the compiler can fold `valid` to true and eliminate branches
    for i in 0..items_per_thread {
        let local_idx = lane_id + i * PLANE_DIM;
        let global_idx = sub_part_start + local_idx;
        let valid = is_full_plane || global_idx < num_items;

        // Clamp index to avoid out-of-bounds read (select doesn't short-circuit on GPU)
        let safe_idx = select(valid, global_idx, 0u32);
        let key = select(valid, KIn::to_radix(keys_in[safe_idx as usize]), max_radix);
        keys[i as usize] = key;

        if has_values {
            values[i as usize] = select(valid, values_in[safe_idx as usize], 0u32);
        }

        let digit_radix = (key >> shift) & digit_mask;
        let digit = u32::cast_from(digit_radix);

        // Warp-level ranking with validity mask
        let peer_mask = compute_peer_mask(digit, valid);
        let rank = count_lower_peers(peer_mask, lane_id);
        let total = count_set_bits(peer_mask);
        let leader = find_first_set_bit(peer_mask);
        let is_leader = lane_id == leader && valid;

        let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
        let mut base = 0u32;
        if is_leader {
            base = plane_hists[hist_idx].fetch_add(total);
        }
        base = plane_shuffle(base, leader);

        local_offsets[i as usize] = base + rank;
    }
    sync_cube();

    // Phase 2: Reduce plane histograms and compute digit starts
    // Each thread handles one digit (first 256 threads)
    // Write prefix sums to non-atomic array for faster reads in Phase 3
    if thread_id < NUM_BUCKETS_U32 {
        let mut sum = 0u32;

        // Sum across all planes for this digit, converting to exclusive prefix within each plane
        for p in 0..num_planes {
            let idx = (p as usize) * NUM_BUCKETS + thread_id as usize;
            let count = plane_hists[idx].load();
            plane_hists[idx].store(sum);
            sum += count;
        }

        // sum is now total count for this digit in this block
        // Store it temporarily in digit_start for the prefix sum
        digit_start[thread_id as usize] = sum;
    }
    sync_cube();

    // Compute exclusive prefix sum across 256 digits using warp-level primitives
    // Fused steps to reduce sync barriers
    #[allow(clippy::manual_div_ceil)]
    let num_digit_warps = (NUM_BUCKETS_U32 + PLANE_DIM - 1) / PLANE_DIM;

    // Step 1: Warp-level exclusive scan and store warp totals
    if thread_id < NUM_BUCKETS_U32 {
        let val = digit_start[thread_id as usize];
        let warp_exclusive = plane_exclusive_sum(val);
        let my_inclusive = warp_exclusive + val;
        let warp_total = plane_shuffle(my_inclusive, PLANE_DIM - 1);

        digit_start[thread_id as usize] = warp_exclusive;

        if lane_id == 0 {
            digit_global[plane_id as usize] = warp_total;
        }
    }
    sync_cube();

    // Step 2 & 3 fused: First warp computes prefix, all threads read and compute final
    // All threads must participate in plane ops (partial participation hangs on CUDA)
    let warp_input = if plane_id == 0 && lane_id < num_digit_warps {
        digit_global[lane_id as usize]
    } else {
        #[allow(clippy::useless_conversion)]
        0u32.into()
    };

    let warp_prefix = plane_exclusive_sum(warp_input);
    if plane_id == 0 && lane_id < num_digit_warps {
        digit_global[lane_id as usize] = warp_prefix;
    }
    sync_cube();

    // All digit threads add warp prefix and compute global offset
    if thread_id < NUM_BUCKETS_U32 {
        let warp_prefix = digit_global[plane_id as usize];
        let my_start = digit_start[thread_id as usize] + warp_prefix;
        digit_start[thread_id as usize] = my_start;

        let block_offset = block_offsets[block_id as usize * NUM_BUCKETS + thread_id as usize];
        digit_global[thread_id as usize] = block_offset - my_start;
    }
    sync_cube();

    // Phase 3: Scatter keys to shared memory (local reordering by digit)
    // Recompute digit from key to reduce register pressure
    // When is_full_plane is true, the compiler can fold `valid` to true and eliminate branches
    for i in 0..items_per_thread {
        let local_idx = lane_id + i * PLANE_DIM;
        let global_idx = sub_part_start + local_idx;
        let valid = is_full_plane || global_idx < num_items;

        if valid {
            let key = keys[i as usize];
            let digit_radix = (key >> shift) & digit_mask;
            let digit = u32::cast_from(digit_radix);
            let offset_in_plane = local_offsets[i as usize];

            let hist_idx = plane_id as usize * NUM_BUCKETS + digit as usize;
            let plane_prefix = plane_hists[hist_idx].load();
            let digit_base = digit_start[digit as usize];
            let local_pos = digit_base + plane_prefix + offset_in_plane;

            shared_keys[local_pos as usize] = key;
            if has_values {
                shared_values[local_pos as usize] = values[i as usize];
            }
        }
    }
    sync_cube();

    // Phase 4: Coalesced read from shared memory and write to global
    // Keys are now grouped by digit, so sequential reads are cache-friendly
    // When is_full_block is true, the compiler can fold `valid` to true and eliminate branches
    let is_full_block = block_start + items_per_block <= num_items;
    let items_in_block = select(num_items > block_start, num_items - block_start, 0u32);

    for i in 0..items_per_thread {
        let local_idx = thread_id + i * CUBE_DIM;
        let valid = is_full_block || local_idx < items_in_block;

        if valid {
            let key = shared_keys[local_idx as usize];
            let digit_radix = (key >> shift) & digit_mask;
            let digit = u32::cast_from(digit_radix);
            let ascending_pos = digit_global[digit as usize] + local_idx;

            // For descending sort on the final pass, reverse the output position
            let global_pos = select(reverse_output, num_items - 1 - ascending_pos, ascending_pos);

            keys_out[global_pos as usize] = KOut::from_radix(key);

            if has_values {
                values_out[global_pos as usize] = shared_values[local_idx as usize];
            }
        }
    }
}
