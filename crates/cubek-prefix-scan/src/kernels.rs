use crate::{
    Blueprint, ScanKind,
    eval::strategy::{CubeScanStrategy, GlobalScanStrategy},
    instructions::ScanInstruction,
};
use cubecl::prelude::*;

#[cube]
pub fn vector_inclusive_prefix_scan<T: Numeric, N: Size, I: ScanInstruction>(
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
pub fn vector_exclusive_prefix_scan<T: Numeric, N: Size, I: ScanInstruction>(
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
pub fn plane_inclusive_prefix_scan<T: Numeric, I: ScanInstruction>(
    mut value: T,
    #[comptime] plane_size_log: u32,
) -> T {
    // Kogge-Stone / Hillis-Steele
    #[unroll]
    for shift in 0..plane_size_log {
        let offset = 1 << shift;
        let below = plane_shuffle(value, UNIT_POS_PLANE - offset);
        if UNIT_POS_PLANE >= offset {
            value = I::combine::<T>(below, value);
        }
    }
    value
}

#[cube]
pub fn plane_exclusive_prefix_scan<T: Numeric, I: ScanInstruction>(
    mut value: T,
    #[comptime] plane_size_log: u32,
) -> T {
    value = plane_inclusive_prefix_scan::<T, I>(value, plane_size_log);
    let below = plane_shuffle(value, UNIT_POS_PLANE - 1);
    select(UNIT_POS_PLANE >= 1, below, I::identity::<T>())
}

#[cube]
fn cube_inclusive_prefix_scan<T: Numeric, I: ScanInstruction>(
    values: &mut Shared<[T]>,
    #[comptime] blueprint: Blueprint,
) {
    #[comptime]
    match blueprint.strategy.cube {
        CubeScanStrategy::KoggeStoneOrHillisSteele =>
        {
            #[unroll]
            for shift in 0..blueprint.cube_scan_size_log {
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
        CubeScanStrategy::LadnerFischer => {
            // Up-sweep phase: Reduction
            #[unroll]
            for shift in 0..blueprint.cube_scan_size_log {
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
            for shift in 1..blueprint.cube_scan_size_log {
                let offset = blueprint.cube_scan_size >> (shift + 1);
                let modulo_mask = (blueprint.cube_scan_size >> shift) - 1;
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
        CubeScanStrategy::BrentKung => {
            // Up-sweep phase: Inclusive prefix scan
            #[unroll]
            for layer in 0..blueprint.plane_sized_layer_count {
                let scale = 1 << (layer * blueprint.plane_size_log);
                // let active_planes = blueprint.cube_scan_size.div_ceil(scale * blueprint.plane_size);
                let active_planes =
                    (blueprint.cube_scan_size + (scale << blueprint.plane_size_log) - 1)
                        >> ((layer + 1) * blueprint.plane_size_log);
                if PLANE_POS < active_planes {
                    let index = UNIT_POS_X * scale + scale - 1;
                    let plane_scan = I::plane_prefix_scan::<T>(
                        values[index as usize],
                        ScanKind::Inclusive,
                        blueprint.plane_size_log,
                    );
                    values[index as usize] = plane_scan;
                }
                sync_cube();
            }
            // Down-sweep phase: Propagation
            #[unroll]
            for layer in 1..blueprint.plane_sized_layer_count {
                let reverse_layer = blueprint.plane_sized_layer_count - layer - 1;
                let scale = 1 << (reverse_layer * blueprint.plane_size_log);
                // let active_planes = blueprint.cube_scan_size.div_ceil(scale * blueprint.plane_size);
                let active_planes =
                    (blueprint.cube_scan_size + (scale << blueprint.plane_size_log) - 1)
                        >> ((reverse_layer + 1) * blueprint.plane_size_log);
                if PLANE_POS > 0 && PLANE_POS < active_planes {
                    let base = values
                        [((PLANE_POS * blueprint.plane_size - 1) * scale + scale - 1) as usize];
                    let index = UNIT_POS_X * scale + scale - 1;
                    let value = values[index as usize];
                    values[index as usize] = select(
                        UNIT_POS_PLANE < blueprint.plane_size - 1,
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
pub(crate) fn single_pass_prefix_scan<T: Numeric, N: Size, I: ScanInstruction>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    output_atomic: &mut Tensor<Atomic<T>>,
    reductions: &mut Tensor<Atomic<T>>,
    task_allocator: &mut Tensor<Atomic<u32>>,
    task_status: &mut Tensor<Atomic<u32>>,
    #[comptime] blueprint: Blueprint,
    #[define(T)] _dtype: ElemType,
) {
    let mut cube_id = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
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
    let vector = input[(cube_id * blueprint.units_per_cube + UNIT_POS_X) as usize];

    // Unit scan
    let (vector, vector_reduction) = match blueprint.scan_kind {
        ScanKind::Exclusive => vector_exclusive_prefix_scan::<T, N, I>(vector),
        ScanKind::Inclusive => vector_inclusive_prefix_scan::<T, N, I>(vector),
    };

    // Cube scan
    let mut cube_scan = Shared::new_slice(blueprint.cube_scan_size as usize);
    let plane_scan: T;
    #[comptime]
    if blueprint.plane_scan_before_cube_scan {
        plane_scan = I::plane_prefix_scan::<T>(
            vector_reduction,
            ScanKind::Exclusive,
            blueprint.plane_size_log,
        );
        if UNIT_POS_PLANE == blueprint.plane_size - 1 {
            cube_scan[PLANE_POS as usize] = I::combine::<T>(plane_scan, vector_reduction);
        }
        sync_cube();
        if PLANE_POS < blueprint.cube_scan_size {
            cube_inclusive_prefix_scan::<T, I>(&mut cube_scan, blueprint.clone());
        }
    } else {
        plane_scan = I::identity::<T>();
        cube_scan[UNIT_POS_X as usize] = vector_reduction;
        sync_cube();
        cube_inclusive_prefix_scan::<T, I>(&mut cube_scan, blueprint.clone());
    }

    // Global scan
    let mut spin_cycles = 2u32;
    let cube_reduction = cube_scan[(blueprint.cube_scan_size - 1) as usize];
    if let GlobalScanStrategy::DecoupledLookback = blueprint.strategy.global {
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
        let mut cube_prefix = I::identity::<T>();
        let mut prev_cube_id = cube_id;
        while prev_cube_id > 0 {
            prev_cube_id -= 1;
            let mut prev_cube_status: u32 = task_status[prev_cube_id as usize].load();
            while prev_cube_status == 0u32 {
                spin_cycles += 1;
                prev_cube_status = task_status[prev_cube_id as usize].load();
            }
            if let GlobalScanStrategy::DecoupledLookback = blueprint.strategy.global {
                if prev_cube_status == 1u32 {
                    cube_prefix =
                        I::combine::<T>(reductions[prev_cube_id as usize].load(), cube_prefix);
                } else {
                    cube_prefix = I::combine::<T>(
                        output_atomic[((prev_cube_id + 1)
                            * blueprint.units_per_cube
                            * input.vector_size() as u32
                            - blueprint.scan_kind as u32)
                            as usize]
                            .load(),
                        cube_prefix,
                    );
                    break;
                }
            } else {
                cube_prefix = output_atomic[((prev_cube_id + 1)
                    * blueprint.units_per_cube
                    * input.vector_size() as u32
                    - blueprint.scan_kind as u32)
                    as usize]
                    .load();
                break;
            }
        }
        cube_scan[(blueprint.cube_scan_size - 1) as usize] = cube_prefix;

        // Global broadcast of prefix
        if cube_id < CUBE_COUNT_X {
            output_atomic[((cube_id + 1) * blueprint.units_per_cube * input.vector_size() as u32
                - blueprint.scan_kind as u32) as usize]
                .store(I::combine::<T>(cube_prefix, cube_reduction));
        }
    }
    sync_storage();
    if UNIT_POS_X == 0 {
        task_status[cube_id as usize].store(spin_cycles);
    }

    // Unit propagation
    let cube_prefix = cube_scan[(blueprint.cube_scan_size - 1) as usize];
    let mut unit_prefix: T;
    #[comptime]
    if blueprint.plane_scan_before_cube_scan {
        unit_prefix = plane_scan;
        if PLANE_POS > 0 {
            let plane_prefix = cube_scan[(PLANE_POS - 1) as usize];
            unit_prefix = I::combine::<T>(plane_prefix, unit_prefix);
        }
    } else {
        unit_prefix = select(
            UNIT_POS_X > 0,
            cube_scan[(UNIT_POS_X - 1) as usize],
            I::identity::<T>(),
        );
    }
    let vector_prefix = I::combine::<T>(cube_prefix, unit_prefix);
    let vector = I::combine_vector::<T, N>(Vector::new(vector_prefix), vector);

    // Vectorized store to global memory
    output[(cube_id * blueprint.units_per_cube + UNIT_POS_X) as usize] = vector;
}
