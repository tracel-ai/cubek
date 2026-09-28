//! Device time of `matmul_mb`'s lane-split contraction, static and dynamic.
//!
//! Not a check: run it alone (`--ignored --exact tile::mb_timing::mb_lanes_timing --nocapture`)
//! in two trees and compare the lines it prints. The kernel is `matmul_mb`'s own body under its two tilings: a
//! weight contiguous along `K`, its plane's lanes taking the steps of a column in turns under a
//! walk over the chunks, and one contiguous along `N`, each lane owning a run of columns and
//! walking the whole of `K` below the lanes. Arms are measured in interleaved rounds, so drift in
//! the device's clocks lands on all of them alike.

use std::sync::Arc;
use std::time::Duration;

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const VECTOR: usize = 4;
const VECTORS_PER_STEP: usize = 4;
const STEP: usize = VECTOR * VECTORS_PER_STEP;
const LANES: usize = 32;
const PLANES: usize = 8;

const K_EXTENT: usize = 4096;
const N_EXTENT: usize = 4096;
const ROUNDS: usize = 4;
const SAMPLES: usize = 30;

/// `matmul_mb_kernel`'s body: a plane's register block over its two levels, lanes and walk in
/// whichever order the tiling puts them, drained through `cells`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn matmul_mb<E: Numeric, VA: Size, VB: Size, VC: Size>(
    x: &TileArg<'_, E, VA>,
    w: &TileArg<'_, E, VB>,
    out: &TileArg<'_, E, VC>,
    partitioning: Partitioning,
    #[comptime] block: RegisterBlock,
    #[comptime] cells: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = x.tile(comptime!(partitioning.clone()));
    let b = w.tile(comptime!(partitioning.clone()));
    let c = out.tile(comptime!(partitioning.clone()));
    for cube in &partitioning {
        for plane in cube {
            let (a_plane, b_plane, c_plane) = (a.at(&plane), b.at(&plane), c.at(&plane));
            let mut sum = c_plane.block_accumulator::<E, E, E>(
                &a_plane,
                &b_plane,
                comptime!(Fragments::below(&c_plane, &a_plane)),
                block,
                Monoid::Sum,
            );
            sum.zero();
            for step in &plane {
                for leaf in step {
                    let mut sum_leaf = sum.at(&leaf);
                    sum_leaf.mma(&a.at(&leaf), &b.at(&leaf), Semiring::SUM_PROD);
                }
            }
            sum.drained_into(&c_plane, comptime!(Some(cells.clone())));
        }
    }
}

/// Which of the weight's axes is contiguous, which decides the tiling.
#[derive(Clone, Copy, Debug)]
enum Weight {
    /// Stored `[n, k]`: a column's lanes take its steps in turns.
    KContiguous,
    /// Stored `[k, n]`: a lane owns a run of columns and walks all of `K`.
    NContiguous,
}

#[derive(Clone, Copy, Debug)]
enum Form {
    Static,
    DynamicN,
    DynamicNK,
}

const FORMS: [Form; 3] = [Form::Static, Form::DynamicN, Form::DynamicNK];

impl Form {
    fn kernel_form(self) -> KernelForm<'static> {
        match self {
            Form::Static => KernelForm::Static,
            Form::DynamicN => KernelForm::DynamicAlong(&[N]),
            Form::DynamicNK => KernelForm::DynamicAlong(&[N, K]),
        }
    }
}

struct Arm {
    client: Client,
    weight: Weight,
    form: Form,
    m: usize,
    k: usize,
    n: usize,
    /// Inputs holding [`x_value`] and [`w_value`] rather than zeros, for a run that is checked.
    filled: bool,
}

impl Arm {
    fn label(&self) -> String {
        format!("{:?} m={} {:?}", self.weight, self.m, self.form)
    }
}

impl Benchmark for Arm {
    type Input = Arc<(TensorHandle, TensorHandle, TensorHandle)>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let (m, k, n) = (self.m, self.k, self.n);
        let dtype = f32::elem_type_native();
        let (w_shape, w_values): (_, Vec<f32>) = match self.weight {
            Weight::KContiguous => (
                shape![n, k],
                (0..n * k).map(|idx| w_value(idx / k, idx % k)).collect(),
            ),
            Weight::NContiguous => (
                shape![k, n],
                (0..k * n).map(|idx| w_value(idx % n, idx / n)).collect(),
            ),
        };
        // The time does not depend on the values, and filling a weight on the host every run is
        // most of a round.
        let input = |shape, values: Vec<f32>| {
            let builder = TestInput::builder(self.client.clone(), shape).dtype(dtype);
            match self.filled {
                true => builder.custom(values).generate_with_f32_host_data().0,
                false => builder.zeros().generate_without_host_data(),
            }
        };
        let x = input(
            shape![m, k],
            (0..m * k).map(|idx| x_value(idx / k, idx % k)).collect(),
        );
        let out = TestInput::builder(self.client.clone(), shape![m, n])
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();
        Arc::new((x, input(w_shape, w_values), out))
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let (x, w, out) = &*input;
        let (m, k, n) = (self.m, self.k, self.n);
        let dtype = f32::elem_type_native();
        let (tiling, block, out_vector) = match self.weight {
            Weight::KContiguous => (
                Tiling::leaf(&[(M, m), (N, 1), (K, STEP)])
                    .lanes(&[(K, LANES)])
                    .interleaved(K)
                    .walk_every(&[K]),
                m * VECTOR,
                1,
            ),
            Weight::NContiguous => (
                Tiling::leaf(&[(M, m), (N, STEP), (K, STEP)])
                    .walk_every(&[K])
                    .lanes(&[(N, LANES)]),
                m * STEP,
                VECTOR,
            ),
        };
        let levels = tiling.planes(&[(N, PLANES)]).cubes(&[N]).levels();
        // The lanes' level, which the drain walks: under the walk where the lanes take `K` in
        // turns, above it where each walks its own.
        let cells = match self.weight {
            Weight::KContiguous => levels[levels.len() - 1].clone(),
            Weight::NContiguous => levels[levels.len() - 2].clone(),
        };
        let partitioning = Partitioning::new(Space::new(&[(M, m), (N, n), (K, k)]), levels);
        let grid = (
            partitioning.cube_count(),
            partitioning.cube_dim(LANES as u32),
        );
        let launcher =
            Launcher::partitioned(&self.client, partitioning, grid, self.form.kernel_form());

        let a = launcher
            .arg(x.clone().binding())
            .subspace(&[M, K])
            .vectorize(VECTOR)
            .build();
        let mut weight = w.clone().binding();
        if let Weight::KContiguous = self.weight {
            weight.shape = [k, n].into();
            weight.strides = [1, k].into();
        }
        let b = launcher
            .arg(weight)
            .subspace(&[K, N])
            .stored()
            .vectorize(VECTOR)
            .build();
        let c = launcher
            .arg(out.clone().binding())
            .subspace(&[M, N])
            .vectorize(out_vector)
            .build();
        matmul_mb::launch(
            &self.client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a.vector_size,
            b.vector_size,
            c.vector_size,
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            RegisterBlock::new(block),
            cells,
            dtype,
        );
        Ok(())
    }

    fn num_samples(&self) -> usize {
        SAMPLES
    }

    fn name(&self) -> String {
        self.label()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "mb-timing", || self.execute(args))
    }
}

fn x_value(r: usize, i: usize) -> f32 {
    ((r * 7 + i) % 5) as f32 - 2.0
}

/// The weight at column `j`, contraction position `i`, whichever way it is stored.
fn w_value(j: usize, i: usize) -> f32 {
    ((j * 3 + i) % 7) as f32 - 3.0
}

/// Every form against the host, at a `k` of two steps per lane where the lanes take `K` in turns.
#[test]
fn every_form_matches_the_host() {
    let client = cubecl::test_device().client();
    let (k, n) = (1024, 4096);
    for weight in [Weight::KContiguous, Weight::NContiguous] {
        for m in [1, 8] {
            let want: Vec<f32> = (0..m * n)
                .map(|idx| {
                    let (r, j) = (idx / n, idx % n);
                    (0..k).map(|i| x_value(r, i) * w_value(j, i)).sum()
                })
                .collect();
            for form in FORMS {
                let arm = Arm {
                    client: client.clone(),
                    weight,
                    form,
                    m,
                    k,
                    n,
                    filled: true,
                };
                let input = arm.prepare();
                arm.execute(input.clone()).unwrap();
                let got = HostData::from_tensor_handle(&client, input.2.clone(), HostDataType::F32);
                let wrong = (0..m * n)
                    .filter(|&idx| (got.get_f32(&[idx / n, idx % n]) - want[idx]).abs() > 1e-3)
                    .count();
                assert_eq!(
                    wrong,
                    0,
                    "{}: {wrong} of {} cells differ",
                    arm.label(),
                    m * n
                );
            }
        }
    }
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i].as_secs_f64() * 1e6
}

#[test]
#[ignore = "timing: run alone with --nocapture"]
fn mb_lanes_timing() {
    let client = cubecl::test_device().client();
    let mut arms = Vec::new();
    for weight in [Weight::KContiguous, Weight::NContiguous] {
        for m in [1, 8] {
            for form in FORMS {
                arms.push((
                    Arm {
                        client: client.clone(),
                        weight,
                        form,
                        m,
                        k: K_EXTENT,
                        n: N_EXTENT,
                        filled: false,
                    },
                    Vec::new(),
                ));
            }
        }
    }
    // A line per arm keeps a remote run from looking idle for the minutes a round takes.
    for round in 0..ROUNDS {
        for (arm, durations) in arms.iter_mut() {
            let run = arm
                .run(TimingMethod::Device)
                .unwrap_or_else(|e| panic!("{}: {e}", arm.label()));
            durations.extend(run.durations);
            eprintln!("round {round}: {}", arm.label());
        }
    }
    println!("k={K_EXTENT} n={N_EXTENT} f32, {ROUNDS} rounds x {SAMPLES} samples, device time");
    for (arm, durations) in arms.iter_mut() {
        durations.sort();
        println!(
            "TIMING {:<48} median {:>8.1} us  p10 {:>8.1}  p90 {:>8.1}",
            arm.label(),
            percentile(durations, 0.5),
            percentile(durations, 0.1),
            percentile(durations, 0.9),
        );
    }
}
