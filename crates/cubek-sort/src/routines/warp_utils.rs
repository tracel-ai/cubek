use crate::routines::RADIX_BITS;
use cubecl::prelude::*;

/// Compute a bitmask of peers in the same warp that have the same digit and are valid.
#[cube]
pub fn compute_peer_mask(digit: u32, valid: bool) -> Line<u32> {
    let mut mask = plane_ballot(valid);

    for k in 0..RADIX_BITS {
        let has_bit = ((digit >> (k as u32)) & 1) != 0;
        let ballot = plane_ballot(has_bit);
        let xor_val = select(has_bit, 0u32, 0xFFFFFFFFu32);

        mask[0] &= ballot[0] ^ xor_val;
        if PLANE_DIM > 32 {
            mask[1] &= ballot[1] ^ xor_val;
        }
        if PLANE_DIM > 64 {
            mask[2] &= ballot[2] ^ xor_val;
        }
        if PLANE_DIM > 96 {
            mask[3] &= ballot[3] ^ xor_val;
        }
    }
    mask
}

/// Count the number of peers with lower lane IDs in the mask.
#[cube]
pub fn count_lower_peers(mask: Line<u32>, lane_id: u32) -> u32 {
    let mut count = 0u32;

    let lt_mask_0 = select(lane_id < 32, (1u32 << lane_id) - 1, 0xFFFFFFFFu32);
    count += (mask[0] & lt_mask_0).count_ones();

    if PLANE_DIM > 32 {
        let lt_mask_1 = select(
            lane_id < 32,
            0u32,
            select(lane_id < 64, (1u32 << (lane_id - 32)) - 1, 0xFFFFFFFFu32),
        );
        count += (mask[1] & lt_mask_1).count_ones();
    }

    if PLANE_DIM > 64 {
        let lt_mask_2 = select(
            lane_id < 64,
            0u32,
            select(lane_id < 96, (1u32 << (lane_id - 64)) - 1, 0xFFFFFFFFu32),
        );
        count += (mask[2] & lt_mask_2).count_ones();
    }

    if PLANE_DIM > 96 {
        let lt_mask_3 = select(lane_id < 96, 0u32, (1u32 << (lane_id - 96)) - 1);
        count += (mask[3] & lt_mask_3).count_ones();
    }

    count
}

/// Count total set bits in the mask.
#[cube]
pub fn count_set_bits(mask: Line<u32>) -> u32 {
    let mut count = mask[0].count_ones();
    if PLANE_DIM > 32 {
        count += mask[1].count_ones();
    }
    if PLANE_DIM > 64 {
        count += mask[2].count_ones();
    }
    if PLANE_DIM > 96 {
        count += mask[3].count_ones();
    }
    count
}

/// Find the lane ID of the first set bit in the mask.
/// Uses min() with trailing_zeros() - returns 32 for zero input, so min naturally picks the first set bit.
#[cube]
pub fn find_first_set_bit(mask: Line<u32>) -> u32 {
    let mut result = mask[0].trailing_zeros();

    if PLANE_DIM > 32 {
        result = u32::min(result, 32 + mask[1].trailing_zeros());
    }
    if PLANE_DIM > 64 {
        result = u32::min(result, 64 + mask[2].trailing_zeros());
    }
    if PLANE_DIM > 96 {
        result = u32::min(result, 96 + mask[3].trailing_zeros());
    }

    result
}
