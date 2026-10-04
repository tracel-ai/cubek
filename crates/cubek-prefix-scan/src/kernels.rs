use crate::{
    base::{PrefixScanGlobalStrategy, PrefixScanKind, PrefixScanLocalStrategy},
    blueprint::PrefixScanBlueprint,
    instructions::PrefixScanInstruction,
};
use cubecl::prelude::*;

#[cube]
pub fn vector_inclusive_prefix_scan<T: Numeric, N: Size, I: PrefixScanInstruction>(
    mut values: Vector<T, N>,
) -> (Vector<T, N>, T) {
    let mut accumulator = values.extract(0usize);
    #[unroll]
    for v in 1..values.vector_size() {
        accumulator = I::combine::<T>(accumulator, values.extract(v));
        values.insert(v, accumulator);
    }
    (values, accumulator)
}

#[cube]
pub fn vector_exclusive_prefix_scan<T: Numeric, N: Size, I: PrefixScanInstruction>(
    mut values: Vector<T, N>,
) -> (Vector<T, N>, T) {
    let mut accumulator = I::identity::<T>();
    #[unroll]
    for v in 0..values.vector_size() {
        let prev_accumulator = accumulator;
        accumulator = I::combine::<T>(accumulator, values.extract(v));
        values.insert(v, prev_accumulator);
    }
    (values, accumulator)
}

#[cube]
pub fn plane_inclusive_prefix_scan<T: Numeric, I: PrefixScanInstruction>(mut value: T) -> T {
    // Kogge-Stone / Hillis-Steele
    let plane_size_ilog2 = u32::find_first_set(PLANE_DIM) - 1;
    for shift in 0..plane_size_ilog2 {
        let offset = 1 << shift;
        if UNIT_POS_PLANE >= offset {
            value = I::combine::<T>(plane_shuffle(value, UNIT_POS_PLANE - offset), value);
        }
    }
    value
}

#[cube]
pub fn plane_exclusive_prefix_scan<T: Numeric, I: PrefixScanInstruction>(mut value: T) -> T {
    value = plane_inclusive_prefix_scan::<T, I>(value);
    select(
        UNIT_POS_PLANE >= 1,
        plane_shuffle(value, UNIT_POS_PLANE - 1),
        I::identity::<T>(),
    )
}

#[cube]
pub fn cube_inclusive_prefix_scan<T: Numeric, I: PrefixScanInstruction>(
    values: &mut Shared<[T]>,
    #[comptime] cube_scan_size: u32,
    #[comptime] strategy: PrefixScanLocalStrategy,
) {
    #[comptime]
    match strategy {
        PrefixScanLocalStrategy::KoggeStoneOrHillisSteele => {
            #[unroll]
            for shift in 0..cube_scan_size.ilog2() {
                let offset = 1u32 << shift;
                let mut value = values[UNIT_POS_X as usize];
                if UNIT_POS_X >= offset {
                    value = I::combine::<T>(values[(UNIT_POS_X - offset) as usize], value);
                }
                sync_cube();
                values[UNIT_POS_X as usize] = value;
                sync_cube();
            }
        }
        PrefixScanLocalStrategy::LadnerFischer => {
            // Up-sweep phase: Reduction
            #[unroll]
            for shift in 0..cube_scan_size.ilog2() {
                let offset = 1u32 << shift;
                let modulo_mask = (2u32 << shift) - 1;
                let active_residue_class = modulo_mask;
                let is_unit_active = UNIT_POS_X & modulo_mask == active_residue_class;
                let mut value = values[UNIT_POS_X as usize];
                if is_unit_active {
                    value = I::combine::<T>(values[(UNIT_POS_X - offset) as usize], value);
                }
                sync_cube();
                if is_unit_active {
                    values[UNIT_POS_X as usize] = value;
                }
                sync_cube();
            }
            // Down-sweep phase: Inclusive prefix scan
            #[unroll]
            for shift in 1..cube_scan_size.ilog2() {
                let offset = cube_scan_size >> (shift + 1);
                let modulo_mask = (cube_scan_size >> shift) - 1;
                let active_residue_class = offset - 1;
                let is_unit_active =
                    (UNIT_POS_X & modulo_mask == active_residue_class) & (UNIT_POS_X >= offset);
                let mut value = values[UNIT_POS_X as usize];
                if is_unit_active {
                    value = I::combine::<T>(values[(UNIT_POS_X - offset) as usize], value);
                }
                sync_cube();
                if is_unit_active {
                    values[UNIT_POS_X as usize] = value;
                }
                sync_cube();
            }
        }
        PrefixScanLocalStrategy::BrentKung => {
            let plane_size_ilog2 = u32::find_first_set(PLANE_DIM) - 1;
            let plane_sized_layer_count = (u32::find_first_set(cube_scan_size) - 1)
                .div_ceil(u32::find_first_set(PLANE_DIM) - 1);
            // Up-sweep phase: Inclusive prefix scan
            for layer in 0..plane_sized_layer_count {
                let scale = 1 << (layer * plane_size_ilog2);
                // let active_planes = cube_scan_size.div_ceil(scale * PLANE_DIM);
                let active_planes = (cube_scan_size + (scale << plane_size_ilog2) - 1)
                    >> ((layer + 1) * plane_size_ilog2);
                if PLANE_POS < active_planes {
                    let index = UNIT_POS_X * scale + scale - 1;
                    let plane_scan = I::plane_prefix_scan::<T>(
                        values[index as usize],
                        PrefixScanKind::Inclusive,
                    );
                    values[index as usize] = plane_scan;
                }
                sync_cube();
            }
            // Down-sweep phase: Propagation
            for layer in 1..plane_sized_layer_count {
                let reverse_layer = plane_sized_layer_count - layer - 1;
                let scale = 1 << (reverse_layer * plane_size_ilog2);
                // let active_planes = cube_scan_size.div_ceil(scale * PLANE_DIM);
                let active_planes = (cube_scan_size + (scale << plane_size_ilog2) - 1)
                    >> ((reverse_layer + 1) * plane_size_ilog2);
                if PLANE_POS > 0 && PLANE_POS < active_planes {
                    let base = values[((PLANE_POS * PLANE_DIM - 1) * scale + scale - 1) as usize];
                    let index = UNIT_POS_X * scale + scale - 1;
                    let value = values[index as usize];
                    values[index as usize] = select(
                        UNIT_POS_PLANE < PLANE_DIM - 1,
                        I::combine::<T>(base, value),
                        value,
                    );
                }
                sync_cube();
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn single_pass_prefix_scan<T: Numeric, N: Size, I: PrefixScanInstruction>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    output_atomic: &mut Tensor<Atomic<T>>,
    reductions: &mut Tensor<Atomic<T>>,
    task_allocator: &mut Tensor<Atomic<u32>>,
    task_status: &mut Tensor<Atomic<u32>>,
    #[comptime] blueprint: PrefixScanBlueprint,
    #[define(T)] _dtype: ElemType,
) {
    let mut cube_id =
        CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y + CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;

    #[comptime]
    if blueprint.use_task_allocator {
        let mut cube_broadcast = Shared::<u32>::new();
        if UNIT_POS_X == 0 {
            *cube_broadcast = task_allocator[0].fetch_add(1u32);
        }
        sync_cube();
        cube_id = *cube_broadcast;
    }

    // Vectorized load from global memory
    let unit_id = cube_id * CUBE_DIM_X + UNIT_POS_X;
    if unit_id * input.vector_size() as u32 >= input.shape(0) as u32 {
        return;
    }
    let vector = input[unit_id as usize];

    // Unit scan
    let (vector, vector_reduction) = match blueprint.scan_kind {
        PrefixScanKind::Exclusive => vector_exclusive_prefix_scan::<T, N, I>(vector),
        PrefixScanKind::Inclusive => vector_inclusive_prefix_scan::<T, N, I>(vector),
    };

    // Cube scan
    let cube_scan_size = 1u32 << blueprint.cube_scan_size_ilog2;
    let mut cube_scan = Shared::new_slice(cube_scan_size as usize);
    cube_scan[UNIT_POS_X as usize] = vector_reduction;
    sync_cube();
    cube_inclusive_prefix_scan::<T, I>(&mut cube_scan, cube_scan_size, blueprint.strategy.cube);

    // Global scan
    let mut spin_cycles = 2u32;
    let cube_reduction = cube_scan[(cube_scan_size - 1) as usize];
    if let PrefixScanGlobalStrategy::DecoupledLookback = blueprint.strategy.global {
        // Global broadcast of reduction
        if UNIT_POS_X == 0 {
            reductions[cube_id as usize].store(cube_reduction);
        }
        sync_storage();
        if UNIT_POS_X == 0 {
            task_status[cube_id as usize].store(1u32);
        }
    }
    if UNIT_POS_X == 0 {
        #[comptime]
        let scan_kind = match blueprint.scan_kind {
            PrefixScanKind::Exclusive => 0,
            PrefixScanKind::Inclusive => 1,
        };

        let mut cube_prefix = I::identity::<T>();
        let mut prev_cube_id = cube_id;
        while prev_cube_id > 0 {
            prev_cube_id -= 1;
            let mut prev_cube_status: u32 = task_status[prev_cube_id as usize].load();
            while prev_cube_status == 0u32 {
                spin_cycles += 1;
                prev_cube_status = task_status[prev_cube_id as usize].load();
            }
            #[comptime]
            if let PrefixScanGlobalStrategy::DecoupledLookback = blueprint.strategy.global {
                if prev_cube_status == 1u32 {
                    cube_prefix =
                        I::combine::<T>(reductions[prev_cube_id as usize].load(), cube_prefix);
                } else {
                    cube_prefix = I::combine::<T>(
                        output_atomic[((prev_cube_id + 1) * CUBE_DIM_X * input.vector_size() as u32
                            - scan_kind) as usize]
                            .load(),
                        cube_prefix,
                    );
                    break;
                }
            } else {
                cube_prefix =
                    output_atomic[((prev_cube_id + 1) * CUBE_DIM_X * input.vector_size() as u32
                        - scan_kind) as usize]
                        .load();
                break;
            }
        }
        cube_scan[(cube_scan_size - 1) as usize] = cube_prefix;

        // Global broadcast of prefix
        let output_index = (cube_id + 1) * CUBE_DIM_X * input.vector_size() as u32 - scan_kind;
        if output_index < input.shape(0) as u32 {
            output_atomic[output_index as usize]
                .store(I::combine::<T>(cube_prefix, cube_reduction));
        }
    }
    sync_storage();
    if UNIT_POS_X == 0 {
        task_status[cube_id as usize].store(spin_cycles);
    }

    // Unit propagation
    let cube_prefix = cube_scan[(cube_scan_size - 1) as usize];
    let unit_prefix = select(
        UNIT_POS_X > 0,
        cube_scan[(UNIT_POS_X - 1) as usize],
        I::identity::<T>(),
    );
    let vector_prefix = I::combine::<T>(cube_prefix, unit_prefix);
    let vector = I::combine_vector::<T, N>(Vector::new(vector_prefix), vector);

    // Vectorized store to global memory
    output[unit_id as usize] = vector;
}
