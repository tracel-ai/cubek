use cubecl::prelude::*;
use cubecl::std::tensor::View;
use std::f32::consts::PI;

use super::{PrngArgs, PrngRuntime, random};

use crate::{RandomFamily, lcg_step, taus_step_0, taus_step_1, taus_step_2, to_unit_interval_open};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Normal {
    mean: f32,
    std: f32,
}

#[derive(Debug)]
struct NormalFamily;

impl RandomFamily for NormalFamily {
    type Runtime = Normal;
}

#[cube]
impl PrngRuntime for Normal {
    fn inner_loop<E: Numeric, N: Size>(
        args: Normal,
        write_index_base: usize,
        n_invocations: u32,
        #[comptime] n_values_per_thread: usize,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut View<Vector<E, N>, usize, ReadWrite>,
    ) {
        let mean = f32::cast_from(args.mean);
        let std = f32::cast_from(args.std);

        let mut output_vector_0 = Vector::empty();
        let mut output_vector_1 = Vector::empty();

        let num_iterations = n_values_per_thread / N::value() / 2;
        #[unroll(num_iterations <= 8)]
        for vector_index in 0..num_iterations {
            // vectorization
            #[unroll]
            for i in 0..N::value() {
                // First random uniform integer
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let unit_0 = to_unit_interval_open(int_random);

                // Second random uniform integer
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let unit_1 = to_unit_interval_open(int_random);

                // Box-Muller transform
                let coeff = unit_0.ln() * -2.0;
                let coeff = coeff.sqrt() * std;
                let trigo_arg = 2.0 * PI * unit_1;

                let normal_0 = f32::cos(trigo_arg) * coeff + mean;
                let normal_1 = f32::sin(trigo_arg) * coeff + mean;

                output_vector_0[i] = E::cast_from(normal_0);
                output_vector_1[i] = E::cast_from(normal_1);
            }

            let iteration_offset = vector_index * n_invocations as usize * 2;
            let write_index_0 = write_index_base + iteration_offset;
            let write_index_1 = write_index_0 + n_invocations as usize;

            output[write_index_0] = output_vector_0;
            output[write_index_1] = output_vector_1;
        }
    }
}

impl PrngArgs for Normal {
    type Args = Self;

    fn args<R: Runtime>(self) -> NormalLaunch<R> {
        NormalLaunch::new(self.mean, self.std)
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: Runtime>(
    client: &ComputeClient<R>,
    mean: f32,
    std: f32,
    out: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    random::<NormalFamily, R>(client, Normal { mean, std }, out, dtype)
}
