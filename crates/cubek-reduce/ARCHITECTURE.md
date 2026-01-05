# Cubek Reduce Architecture

This document describes the architecture of the `cubek-reduce` crate, specifically designing the blueprint system, routines, and launching procedure to optimize JIT compilation and kernel performance.

## Core Philosophy

The central design principle is to **minimize the JIT compilation key (Blueprint)**. By decoupling runtime-variable settings from the compilation key, we prevent "kernel explosion" (compiling a unique kernel for every variation of input size) while maintaining the performance benefits of specialized kernels.

## 1. The Blueprint System

The `ReduceBlueprint` is the compilation key. It dictates the static code generation structure of the kernel.

### Minimal Compilation Key
To ensure efficient caching, the Blueprint only contains information that fundamentally alters the kernel's control flow or logic structure.

**Included in Blueprint:**
*   **Line Mode**: `Parallel` (contiguous memory access) or `Perpendicular` (strided access). This changes the reading logic significantly.
*   **Global Strategy**: High-level approach (`Unit`, `Plane`, or `Cube`).
*   **Structural Flags**:
    *   `use_planes`: Whether to use warp/subgroup primitives.
    *   `bound_checks`: Strategies for handling out-of-bounds access (e.g., `Mask`, `Branch`, `None`).
    *   `idle_mode`: How to handle idle threads.
    *   `independent`: Whether units in a plane work independently.
    *   `num_shared_accumulators`: (For Cube strategy) Shared memory usage structure.

**Excluded from Blueprint (Implicitly in Key):**
These settings significantly affect the kernel but are excluded from the `ReduceBlueprint` struct to avoid duplication and hashing overhead, as they are *already* part of the underlying compilation key:
*   **Line Size (Vectorization)**: The line size changes the input/output tensor types (e.g., `f32` vs `vector<f32, 4>`). Since tensor types are fundamental to the kernel signature, the line size is already captured in the JIT key. Adding it to the blueprint would be redundant.
*   **Cube Dimensions**: The workgroup size (`cube_dim`) is part of the autotune key and kernel configuration. A different `cube_dim` naturally results in a different kernel compilation.

**Runtime Parameters (Not in Key):**
*   **Cube Count**: Calculated dynamically based on input size; does not trigger recompilation.
*   **Problem Dimensions**: `vector_size` and `vector_count` are passed as arguments, allowing one kernel to handle various sizes.

## 2. The Routine System

Routines (`UnitRoutine`, `PlaneRoutine`, `CubeRoutine`) are the architectural bridge between the abstract problem and the concrete kernel configuration.

### Adaptation over Decision
A key architectural choice is that **Routines do not decide vectorization**. They **adapt** to it.

1.  **Line Size Determination**: The launch logic (or autotuner) determines the optimal line size (vectorization) based on input tensor strides and hardware capabilities *before* the routine is involved.
2.  **Routine Adaptation**: The routine receives this line size in `ReduceLineSettings`. It then calculates how to best perform the reduction given that vectorization (e.g., adjusting the number of threads needed, determining if bound checks are now required).

### Routine Preparation Order
The preparation flow is strict:

1.  **Input Analysis**: Determine `LineMode` from tensor strides.
2.  **Vectorization**: Calculate `line_size_input` and `line_size_output`.
3.  **Routine Settings Creation**: Assemble `ReduceLineSettings`.
4.  **Blueprint Generation**: Call `Routine::prepare(..., settings, ...)`.
    *   The routine uses the settings to generate the `ReduceBlueprint` (comptime).
    *   It also calculates the `ReduceLaunchSettings` (runtime: `cube_dim`, `cube_count`).

## 3. Launching Procedure

The launch sequence clearly separates always-executed launch code from cached compilation artifacts.

### 1. Launch Code (Executed Every Time)
These steps happen on the host for every kernel launch:
*   **Input Validation**: Checking shapes and types.
*   **Line Setting Calculation**: Calling `generate_line_size`.
*   **Strategy Selection**: Picking the `RoutineStrategy`.
*   **Routine Preparation**: Generating the blueprint and launch settings.
*   **Kernel Dispatch**: Calling `reduce_kernel::launch_unchecked`.

### 2. Cached Artifacts (JIT Compilation)
The `reduce_kernel` uses the `blueprint` as a `#[comptime]` argument.
*   **Cache Key**: `(Blueprint, ReduceOperationConfig)`.
*   **Comptime Logic**: All logic inside the kernel guarded by `comptime!` or derived from the blueprint runs only once per unique key.

## 4. Validation & Autotuning

Validation is strategically placed within the **comptime** functions (or the `prepare` phase that leads to blueprint generation).

*   **Comptime Validation**: Instead of checking for invalid configurations (like "too much shared memory for this block size") at runtime, we enforce these constraints during the blueprint/kernel generation.
*   **Autotuner Integration**: If a set of parameters results in an invalid blueprint/kernel, it generates a **compilation error** (or a preparation failure). The autotuner catches this and simply marks that configuration as invalid, trying the next one. This avoids runtime crashes and allows the system to aggressively try optimizations that might only work on specific hardware.

## 5. Architecture Diagram

The following diagram illustrates how inputs are filtered and transformed into the minimal compilation key.

```
                                      +-----------------+
                                      | Runtime Inputs  |
                                      | (Shapes, Args)  |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | Launch Logic    |  <-- Always Executed
                                      | (Host Code)     |
                                      +--------+--------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
          +-------------------+                             +--------------------+
          | Vectorization     |                             | Strategy Selection |
          | (Line Size Calc)  |                             | (Unit/Plane/Cube)  |
          +----------+--------+                             +----------+---------+
                     |                                                 |
                     v                                                 v
          +-------------------+                             +--------------------+
          | Routine Settings  |                             | Routine Adaptation |
          | (LineMode, etc.)  +---------------------------->| (Prepare Method)   |
          +-------------------+                             +----------+---------+
                                                                       |
                                                                       v
                                                             +--------------------+
                                                             | ReduceBlueprint    |
                                                             | (Minimal JIT Key)  |
                                                             +---------+----------+
                                                                       |
+----------------------+                                               |
| Implicit JIT Key     |                                               |
| (Tensor Types, Dims) | <---------------------------------------------+
+----------+-----------+
           |
           v
   +----------------+
   | CubeCL Compiler|
   | (Cache Hit?)   |
   +-------+--------+
           |
           v
   +----------------+
   | Optimized      |
   | Kernel Binary  |
   +----------------+
```

## 6. Replicating this Architecture

This "Blueprint + Routine" pattern is a high-performance best practice for `cubecl` kernels and can be replicated for other operations (e.g., Matmul, Convolution).

### Key Takeaways for New Kernels:
1.  **Separate Structure from Parameters**: Identify what changes the *code structure* (Blueprint) vs what just changes the *data* (Runtime Args).
2.  **Implicit Keys**: Remember that function signatures (types) and workgroup dimensions are already part of the unique key. Don't duplicate them in your custom structs.
3.  **Adapt, Don't Decide**: Let the high-level logic or autotuner decide hardware-specifics (like vectorization), and write your routines to *adapt* to those decisions rather than enforcing them.
4.  **Fail Fast at Comptime**: Use `comptime!` validation to reject invalid configs early, allowing the autotuner to skip them cleanly.