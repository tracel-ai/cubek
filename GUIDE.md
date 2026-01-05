# Cubek Kernel Development Guide

This guide outlines the recommended workflow and architectural patterns for creating high-performance kernels in `cubek`. It focuses on making correct architectural decisions regarding the **Blueprint** system and **Routine** adaptation, ensuring efficient JIT compilation and kernel performance.

## The Blueprint-Routine Architecture

The core philosophy of `cubek` is the strict separation of **Kernel Structure** (Compile Time) from **Execution Parameters** (Runtime).

*   **Blueprint**: Represents the minimal set of information required to generate the kernel code. It acts as the unique identifier for JIT compilation.
*   **Routine**: Contains the logic that adapts a generic algorithm to specific hardware constraints, such as vectorization factors or block sizes.
*   **Autotuner**: Responsible for finding the best combination of Routine strategies for a given problem.

## 1. Analyzing the Operation

Before implementing a kernel, it is essential to map out the problem space. Consider how the operation can be parallelized (threads, warps, or blocks), the nature of memory access (contiguous vs. strided), and which hardware primitives (like subgroup operations or shared memory) are applicable.

## 2. Designing the Blueprint (The JIT Key)

The `Blueprint` serves as the JIT compilation key. A unique blueprint results in a unique compiled kernel. To prevent "kernel explosion"—where too many variations of a kernel are compiled—the blueprint must be kept minimal.

**What to Include in the Blueprint**
The blueprint should only contain information that fundamentally alters the control flow or the specific instructions used within the kernel. This includes:
*   **Algorithm Variants**: High-level strategies (e.g., Global vs. Tile vs. Batch).
*   **Hardware Features**: Switches for using specific hardware capabilities like planes (subgroups) or cooperative groups.
*   **Safety Logic**: Strategies for handling boundary conditions, such as using masks versus branching.

**What to Exclude**
Information that is already captured by the kernel signature or runtime arguments should be excluded to avoid duplication.
*   **Vectorization (Line Size)**: The vectorization factor is reflected in the tensor input types (e.g., `vector<f32, 4>`). Including it in the blueprint would duplicate data already present in the JIT key.
*   **Workgroup Dimensions**: The `CubeDim` is already part of the compilation key.
*   **Problem Sizes**: Dimensions like tensor shapes should be passed as runtime arguments.

## 3. Implementing the Routine System

Routines should not make hard decisions about hardware specifics; instead, they should **adapt** to them.

**The Adaptation Workflow**
The launch logic or autotuner determines the optimal constraints (such as vectorization) based on the hardware and input strides. The routine then receives these settings and calculates how to map the algorithm to them.

For example, if the launch logic mandates a line size of 4, the routine does not decide this. Instead, it calculates the necessary thread counts and verifies shared memory usage to accommodate a vectorization of 4. This results in the generation of a `Blueprint` for the compiler and `LaunchSettings` for the runtime.

## 4. Kernel Implementation

The kernel entry point should rely on the blueprint for structural logic. It is recommended to derive a comprehensive configuration object inside the kernel using a `comptime` block. This process acts as "uncompressing" the minimal blueprint, combined with implicit information like line size and cube dimensions, into an easy-to-use structure.

**Example Kernel Signature**

```rust
#[cube(launch_unchecked)]
pub fn my_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] blueprint: MyBlueprint,
) {
    // Expand the blueprint and implicit constants into a usable config
    let config = comptime! {
        // Retrieve implicit information from types and constants
        let line_size = input.line_size(); 
        let cube_dim = CubeDim::default();
        
        // Create a derived configuration struct for internal use
        MyKernelConfig::new(blueprint, line_size, cube_dim)
    };

    // 1. Comptime Validation
    // Validate the expanded config to fail fast if the combination is invalid
    if comptime!(config.requires_planes && !config.hardware_supports_planes) {
        compile_error!("Hardware does not support planes for this configuration");
    }

    // 2. Execution
    // Use the derived config for code generation
    match comptime!(config.strategy) {
        Strategy::A => execute_strategy_a(input, output, config),
        Strategy::B => execute_strategy_b(input, output, config),
    }
}
```

This pattern ensures that the external interface remains clean and the compilation key remains minimal, while the internal implementation benefits from a rich, fully resolved configuration structure.
