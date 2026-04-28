# cubek-test-utils

Shared building blocks for kernel tests in CubeK: test-tensor builders,
host-side reference comparisons, and a unified renderer that pretty-prints
tensors (or diffs them) under a single config.

---

## Configuration: `cubek.toml`

There is **one** place to configure test behavior: a `cubek.toml` file at
the workspace root. The file is read once at process start; the loader
walks up from the current working directory until it finds it.

The shipped `cubek.toml` documents every field with comments. Two sections,
nothing more:

```toml
[test]
policy = "correct"   # "correct" | "strict" | "fail-if-run"

[print]
enabled = false       # toggle all printing
view = "table"        # "table" | "lines"
force-fail = true     # fail every test that prints, so cargo shows stdout
fail-only = false     # diff: only render cells where Δ > ε
show-expected = false # diff: render `got/expected` per cell (else just `got`)
filter = ""           # per-axis filter, same DSL as the slice helper
```

**The whole pipeline obeys one rule:** if `enabled = false`, nothing prints.
Set it to `true`, run a test, watch your tensors render. That's it.

### `[test] policy`

| Policy        | No error | Numerical error | Compilation error |
| ------------- | -------- | --------------- | ----------------- |
| `correct`     | accept   | fail            | accept            |
| `strict`      | accept   | fail            | fail              |
| `fail-if-run` | fail     | accept          | accept            |

If `[print] force-fail = true`, every passing test that printed is
additionally rejected — useful so cargo surfaces the dump (it otherwise
swallows stdout from passing tests). Compile errors are also rejected when
`force-fail = true`, regardless of policy.

---

## Rendering: one path for everything

Both `assert_equals_approx(actual, expected, ε)` and the free
`print_tensors(label, &[&a, &b], Some(ε))` go through the same renderer.
There is no "diff path" vs "pretty-print path"; comparing actual-vs-expected
and pretty-printing two unrelated same-shape tensors are literally the same
call.

Rules:

- One tensor → just values, no color.
- Two tensors of the **same rank and shape** → cells colored green
  (`Δ ≤ ε`) or red (`Δ > ε`). With `show-expected = true` the cell shows
  `got/expected`; otherwise just `got`.
- Two tensors of **different rank or shape** → silently skipped. The
  renderer never panics on bad input.
- Filter rank ≠ tensor rank → silently skipped.

```rust
use cubek_test_utils::print_tensors;

// Single tensor — table or lines per [print] view, no color.
print_tensors("input", &[&host], None);

// Two tensors — colored diff. Same path used by assert_equals_approx.
print_tensors("a vs b", &[&a, &b], Some(1e-3));
```

The table view never shows Δ/ε numbers (cell color carries the info). The lines view always shows
them.

### Table view example (with `show-expected = true`)

```
=== diff  shape=[2, 3] ===
    |                 0                 1                 2
----+------------------------------------------------------
  0 | 0.000000/0.000000 1.000000/1.000000 2.000000/2.000000   ← green
  1 | 4.000000/3.000000 5.000000/4.000000 6.000000/5.000000   ← red
```

### Table view + `fail-only = true`

```
=== diff  shape=[2, 3] ===
    |        0        1        2
----+---------------------------
  0 |                            ← matching cells blanked out
  1 | 4.000000 5.000000 6.000000 ← red
```

### Lines view + `fail-only = true`

```
=== diff  shape=[2, 3] ===
 index |      got | expected |        Δ |        ε | status
-----------------------------------------------------------
[1, 0] | 4.000000 | 3.000000 | 1.000000 | 0.003000 | FAIL    ← red
[1, 1] | 5.000000 | 4.000000 | 1.000000 | 0.004000 | FAIL    ← red
[1, 2] | 6.000000 | 5.000000 | 1.000000 | 0.005000 | FAIL    ← red
```

---

## Filter syntax

Used by both `[print] filter` and `assert_equals_approx_in_slice`. A
comma-separated list of dim entries:

- `.` — wildcard (any index along that dim)
- `N` — a single index
- `M-K` — inclusive range

Example for a 4-D tensor: `.,.,10-20,30` selects all elements where
dim 2 is in `10..=20` and dim 3 is exactly `30`. Filter rank must equal
tensor rank.

From Rust:

```rust
use cubek_test_utils::{DimFilter, assert_equals_approx_in_slice};

// Vec<Range<usize>> works (half-open, like Rust slices).
assert_equals_approx_in_slice(&actual, &expected, 0.001, vec![0..1, 0..3]);

// Or build the canonical TensorFilter explicitly.
let filter = vec![
    DimFilter::Exact(0),
    DimFilter::Range { start: 0, end: 2 }, // inclusive: 0..=2
];
assert_equals_approx_in_slice(&actual, &expected, 0.001, filter);
```

`parse_tensor_filter("0,0-2")` parses the string DSL into a `TensorFilter`.

---

## Failure messages

`assert_equals_approx` collects up to **8** mismatches plus aggregate
stats and reports them in the test panic message:

```
Test failed: Got incorrect results: 17/4096 elements mismatched
  (max |Δ|=0.014648, mean |Δ|=0.004112, worst at [3, 12]) — shape=[16, 256]
First mismatches:
  [0, 5]: got 1.234, expected 1.220, |Δ|=0.014 > ε=0.001
  ...
  ... and 9 more
```

When printing is enabled the per-element output goes to stdout; the panic
message keeps only the aggregate header so it doesn't duplicate the dump.

---

## Test suites

Four suites are available:

- **Light** — tractable subset that runs on CI.
- **Basic** — basic tests that may hang on CI (slow on CPU).
- **Extended** — auto-generated combinatorial tests, kept tractable.
- **Full** — all generable combinations, may not fit.

```bash
# Replace <runtime> with cpu, cuda, rocm, wgpu, vulkan or metal.
cargo test-<runtime>             # basic suite (light on cpu)
cargo test-<runtime>-extended
cargo test-<runtime>-full
```

---

## Building test inputs

Two equivalent ways to construct a test tensor:

```rust
use cubek_test_utils::{TestInput, StrideSpec, DataKind, Distribution};

// Long-form constructor.
let (handle, host) = TestInput::new(
    client.clone(),
    [4, 4],
    f32::as_type_native_unchecked().storage_type(),
    StrideSpec::RowMajor,
    DataKind::Random {
        seed: 0,
        distribution: Distribution::Uniform(-1.0, 1.0),
    },
)
.generate_with_f32_host_data();

// Fluent builder — `dtype` defaults to f32, `stride` defaults to RowMajor.
let (handle, host) = TestInput::builder(client.clone(), [4, 4])
    .uniform( 0, -1.0, 1.0)
    .generate_with_f32_host_data();
```

Builder setters (all optional):

| Setter          | Default                | Effect                      |
| --------------- | ---------------------- | --------------------------- |
| `.dtype(d)`     | `f32`                  | Override the input dtype.   |
| `.stride(spec)` | `StrideSpec::RowMajor` | Override the stride layout. |

Builder finalizers (each returns a `TestInput` ready to generate):

| Finalizer                  | Equivalent `DataKind`                                            |
| -------------------------- | ---------------------------------------------------------------- |
| `.arange()`                | `Arange { scale: None }`                                         |
| `.arange_scaled(s)`        | `Arange { scale: Some(s) }`                                      |
| `.eye()`                   | `Eye`                                                            |
| `.zeros()`                 | `Zeros`                                                          |
| `.uniform(seed, lo, hi)`   | `Random { Uniform(lo, hi) }`                                     |
| `.bernoulli(seed, p)`      | `Random { Bernoulli(p) }`                                        |
| `.normal(seed, mean, std)` | `Random { Normal { mean, std } }`                                |
| `.random(seed, dist)`      | `Random { dist }`                                                |
| `.linspace(start, end)`    | `Custom { data }` with N evenly-spaced values from `start..=end` |
| `.custom(data)`            | `Custom { data }`                                                |

After a finalizer, call any of: `.generate()`, `.generate_with_f32_host_data()`,
`.generate_with_bool_host_data()`, `.generate_test_tensor()`,
`.f32_host_data()`, `.bool_host_data()`.
