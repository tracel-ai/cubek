//! Key transformation kernels for radix sort.
//!
//! Radix sort requires unsigned integer keys. These kernels transform
//! signed integers and floats into a sortable unsigned representation,
//! and transform them back after sorting.

use cubecl::prelude::*;

/// Transform i32 keys to sortable u32 representation.
/// Maps: i32::MIN -> 0, -1 -> 0x7FFFFFFF, 0 -> 0x80000000, i32::MAX -> 0xFFFFFFFF
#[cube(launch_unchecked)]
pub fn transform_i32_to_radix(input: &Tensor<u32>, output: &mut Tensor<u32>, num_items: u32) {
    let idx = CUBE_POS_X * CUBE_DIM + UNIT_POS_X;
    if idx < num_items {
        let value = input[idx as usize];
        // Flip sign bit
        output[idx as usize] = value ^ 0x8000_0000u32;
    }
}

/// Transform sortable u32 representation back to i32 keys.
#[cube(launch_unchecked)]
pub fn transform_radix_to_i32(input: &Tensor<u32>, output: &mut Tensor<u32>, num_items: u32) {
    let idx = CUBE_POS_X * CUBE_DIM + UNIT_POS_X;
    if idx < num_items {
        let value = input[idx as usize];
        // Flip sign bit back
        output[idx as usize] = value ^ 0x8000_0000u32;
    }
}

/// Transform f32 keys to sortable u32 representation.
/// Ensures: -inf < negative < -0 < +0 < positive < +inf < NaN
#[cube(launch_unchecked)]
pub fn transform_f32_to_radix(input: &Tensor<u32>, output: &mut Tensor<u32>, num_items: u32) {
    let idx = CUBE_POS_X * CUBE_DIM + UNIT_POS_X;
    if idx < num_items {
        let bits = input[idx as usize];
        // If sign bit is set (negative), flip all bits
        // If sign bit is clear (positive), flip only sign bit
        let mask = ((bits as i32) >> 31) as u32 | 0x8000_0000u32;
        output[idx as usize] = bits ^ mask;
    }
}

/// Transform sortable u32 representation back to f32 keys.
#[cube(launch_unchecked)]
pub fn transform_radix_to_f32(input: &Tensor<u32>, output: &mut Tensor<u32>, num_items: u32) {
    let idx = CUBE_POS_X * CUBE_DIM + UNIT_POS_X;
    if idx < num_items {
        let value = input[idx as usize];
        // Reverse the transformation
        let mask = (((value >> 31) as i32) - 1) as u32 | 0x8000_0000u32;
        output[idx as usize] = value ^ mask;
    }
}
