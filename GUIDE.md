# Cubek Kernel Development Guide

This guide outlines the recommended workflow and architectural patterns for creating high-performance kernels in `cubek`.
It focuses on helping make key decisions, especially with respect to kernel arguments and whether **runtime** or **compile-time** should be used.

## The Blueprint-Routine Architecture

The core philosophy of `cubek` is the strict separation of kernel structure (Compile Time) from execution parameters.

- **Blueprint**: Represents the minimal set of information required to generate the kernel code.
  A different blueprint will retrigger JIT compilation, serving as compile-time specialization settings.
- **Routine**: Contains the logic that adapts a generic algorithm to specific hardware constraints, such as vectorization factors or cube dimensions.
- **Autotuner**: Responsible for finding the best combination of routine strategies for a given problem.

## Designing the Blueprint

The `Blueprint` serves as the compile-time specialization setting.
A unique blueprint results in a unique compiled kernel.
To prevent kernel explosion, where too many variations of a kernel are compiled, the blueprint must be kept minimal.

### What to Include in the Blueprint

The blueprint should only contain information that fundamentally alters the control flow or the specific instructions used within the kernel.
This includes:

- **Algorithm Variants**: Can be an `enum` that lists all possible ways of executing an algorithm.
- **Algorithm Settings**: Each algorithm can have its own comptime settings that can define loop unrolling, stage size, instructions, etc.
- **Safety Logic**: Strategies for handling boundary conditions, such as using masks versus branching to avoid out-of-bounds access.

### What to Exclude

Information that is already captured by the kernel signature or runtime arguments:

- **Vectorization (Vector Size)**: The vectorization factor is reflected in the tensor input types.
  Including it in the blueprint would duplicate data already present in the JIT key.
- **Cube Dimensions**: The `CubeDim` is already part of the compilation key.
- **Hardware Properties**: Hardware properties can be accessed directly within the kernel, no need to pass them in the blueprint.
- **Problem Sizes**: Dimensions like tensor shapes and strides should be passed as runtime arguments.

## Implementing the Routine System

Routines should not make hard decisions about hardware specifics, instead they should adapt to them.

**The Adaptation Workflow**
The launch logic determines the optimal constraints like vectorization based on the hardware and input shape/strides.
The routine then receives these settings and calculates how to map the algorithm to them.
For example, if the launch logic mandates a vector size of 4, the routine does not decide this.
Instead, it calculates the necessary `cube_dim` and `cube_count` to fully solve the problem.
This results in the generation of a `Blueprint` for the compiler and `LaunchSettings` for the runtime.

## Kernel Implementation

The kernel entry point should rely on the blueprint for structural logic.
You can derive a comprehensive configuration type inside the kernel using a `comptime` block.

This process acts as "uncompressing" the minimal blueprint, combined with implicit information like vector size and hardware properties, into an easy-to-use structure.

### Example Kernel Signature

```rust
#[cube(launch)]
pub fn my_kernel<F: Float>(
    input: &Tensor<Vector<F>>,
    output: &mut Tensor<Vector<F>>,
    #[comptime] blueprint: MyBlueprint,
) {
    let vector_size = input.vector_size();
    let device_properties = comptime::device_properties();

    let config = comptime! {
        // Create a derived configuration struct for internal use
        MyKernelConfig::new(blueprint, vector_size, device_properties)
    };

    // 1. Comptime Validation
    // A comptime assert fires at expand, on the host thread, as a backstop.
    // The real gate is the routine's blueprint/validate step, which rejects an
    // invalid plan before anything is launched: a kernel-side runtime assert
    // fires on a device thread, where it reads as zeroed output, not a rejection.
    comptime!(assert!(
        !(config.requires_planes && !config.hardware_supports_planes),
        "hardware does not support planes for this configuration"
    ));

    // 2. Execution
    // Use the derived config for code generation
    match config.strategy {
        Strategy::A => execute_strategy_a(input, output, config),
        Strategy::B => execute_strategy_b(input, output, config),
    }
}

```

This pattern ensures that the external interface remains clean and the compilation key remains minimal, while the internal implementation benefits from a rich, fully resolved configuration structure.

## The Tile Launch Surface

Tile-DSL kernels are the concrete embodiment of the rules above, with one canonical shape:

```rust
#[cube(launch)]
pub fn my_tile_kernel<E: Numeric, VA: Size, VB: Size>(
    a: &Tensor<Vector<E, VA>>,
    out: &Tensor<Vector<E, VB>>,
    #[comptime] space: Space,
    #[comptime] spec_a: TileSpec,
    #[comptime] spec_out: TileSpec,
) {
    let a = Tile::<E>::of(a, comptime!(space.clone()), spec_a);
    let mut out = Tile::<E>::of(out, space, spec_out);
    // ...
}
```

- **One `Space` per kernel.** The whole iteration space, with its partitioning, passed once.
  Each operand's `TileSpec` carries only what is per-operand: the axes it spans and its `Storage`.
  `Tile::of` projects the space onto those axes in-kernel, so operands cannot disagree about extents or partitioning.
- **Problem sizes stay runtime.** The `Launcher` ships the space with `Extent::Dynamic` top-level extents, resolved in-kernel from the tensor's own shape.
  What forks the compiled kernel is structure only: the partitioner's cuts, each spec's axes, the widths.
- **Width rides the element type.** `Vector<E, V>` with `V: Size` fed a launch value, exactly the vectorization rule above; never a blueprint field.
- **The launch side ships plain tensors.** `launcher.arg(binding).subspace(..).batches(..).vectorize(v).build()` yields a `TensorArg` plus the comptime `TileSpec`; no wrapper struct rides the dispatch, so the tile surface launches at raw-tensor cost.
- **Quantized operands are ordinary tensors too.** The storage-typed values tensor, the scales as a plain second tensor, and the comptime `QuantScheme` (`Tile::of_dequant`); the `Size` launch value for a packed operand is `StridedOperand::bound_width()`.
