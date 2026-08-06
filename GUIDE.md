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
    a: &TileArg<'_, E, VA>,
    out: &TileArg<'_, E, VB>,
    #[comptime] space: Space,
) {
    let a = a.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    // ...
}
```

- **One `Space` per kernel.** The whole iteration space, with its partitioning, passed once.
  Each operand is a [`TileArg`]: its tensor bundled with the comptime `TileSpec` naming only what is per-operand, the axes it spans and its `Storage`.
  `tile(space)` projects the one space onto those axes in-kernel, so operands cannot disagree about extents or partitioning, and a tensor can never pair with another operand's spec.
- **Problem sizes stay runtime.** The `Launcher` ships the space with `Extent::Dynamic` top-level extents, resolved in-kernel from the tensor's own shape.
  What forks the compiled kernel is structure only: the partitioner's cuts, each spec's axes, the widths.
- **Width rides the element type.** The arg's tensor is `Tensor<Vector<E, V>>` with `V: Size` fed a launch value, exactly the vectorization rule above; never a blueprint field.
- **The wrapper is comptime-thin.** `launcher.arg(binding).subspace(..).batches(..).vectorize(v).build().arg()` yields a `TileArgLaunch`: one ordinary tensor binding plus hashed comptime data, so it launches at raw-tensor cost (proven by compiled-kernel diff).
- **A quantized operand is one thing.** It rides its own carrier, `QuantTileArg` (values + scales + comptime spec + scheme), served by `tile::<O>(space)`; a distinct total type, not an option inside `TileArg`. The builder's typestate decides which you get: `.quantized(..)` flips `build()` from `StridedOperand` to `QuantOperand`, whose `arg()` yields the carrier. The `Size` launch value for a packed operand is `QuantOperand::bound_width()`.
