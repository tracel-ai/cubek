# cubek-resample

The `cubek-resample` crate provides a flexible and efficient tensor resampling framework. It enables interpolation, downsampling, upsampling, and various forms of filtering using customizable kernels, boundaries, and algebraic semirings.

## Core Operations

The resampling process is configured using two main structures that group the various definition modules together.

### `Resample`
The main configuration struct for the entire resampling operation. It is composed of:
- **`resample_axes`**: A list of `ResampleAxis` configurations to apply.
- **`semiring`**: The mathematical algebra for the operation.
- **`boundary`**: How out-of-bounds coordinates are handled globally.
- **`normalization`**: How tap weights are normalized across the operation.

### `ResampleAxis`
Configuration for a specific axis being resampled. It is composed of:
- **`axis`**: The target axis index.
- **`kernel`**: The filter shape/distribution used for this axis.
- **`placement`**: The coordinate mapping strategy.
*(Note: Per-axis window sizing and dilation are handled via `WindowArgs` at runtime).*

---

## Definition Modules Breakdown

### Semiring
Determines the mathematical algebra used to combine values and their corresponding weights.

| Semiring | Combine (`f(v, w)`) | Accumulate (`f(acc, v)`) |
| :--- | :--- | :--- |
| **Linear** | `v * w` | `acc + v` |
| **Tropical** | `v + w` | `max(acc, v)` |
| **Log** | `v + w` | `log(exp(acc) + exp(v))` |

### Boundary
Defines how out-of-bounds source coordinates (taps) are handled.

| Mode | Behavior |
| :--- | :--- |
| **Zero** | Out-of-bounds taps contribute 0 |
| **Clamp** | Coordinates are clamped to the nearest valid input |

### Kernel
Specifies the shape and distribution of the resampling filter.

| Kernel | Description |
| :--- | :--- |
| **Zero** | Always returns 0 |
| **Uniform** | Uniform distribution scaled by a factor |
| **Linear** | Triangle kernel (support 2) |
| **Cubic** | Cubic convolution (e.g., Catmull-Rom, Sharp) |
| **Lanczos** | Sinc-sinc function with configurable side-lobes |

### Normalization
Specifies how tap weights are normalized across the resampling window.

| Mode | Behavior |
| :--- | :--- |
| **None** | Preserve kernel weights exactly |
| **Renormalize** | Divide weights by total accumulated valid weight |

### Placement
Determines how output indices map back to source coordinates.

| Placement | Mapping Equation |
| :--- | :--- |
| **Continuous** | `scale * pos + offset` |
| **Windowed** | `step * pos - padding` |

### Internal
- **`accumulator`**: Low-level structures (`Accumulator`, `Value`) used to manage tap accumulation during the core loop.

---

## Practical Examples

To better understand how these definition modules come together, here are practical examples of how `cubek-resample` is configured for actual operations.

### Example: Interpolation
When performing tensor interpolation (like bilinear or bicubic resizing), the modules are typically configured as follows:
- **`Semiring`**: `Linear` (standard mathematical weighted sum)
- **`Boundary`**: `Clamp` (or `Zero` when using the Lanczos kernel)
- **`Kernel`**: Depends directly on the interpolation mode (e.g., `Linear`, `Cubic`, `Lanczos`)
- **`Normalization`**: `None` (or `Renormalize` when using the Lanczos kernel)
- **`Placement`**: `Continuous` (commonly used to handle unaligned grids and scaling)

### Example: Pooling
Resampling can also express pooling operations by swapping the underlying algebra and kernels:
- **Average Pooling**: Uses the `Linear` semiring, a `Uniform` kernel, and `Renormalize` normalization.
- **Max Pooling**: Uses the `Tropical` semiring (which leverages `max()` accumulation) and a `Uniform` kernel.
