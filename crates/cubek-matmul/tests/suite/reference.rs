use cubecl::TestRuntime;
use cubecl::std::tensor::TensorHandle;
use cubecl::{CubeElement, client::ComputeClient};
use cubek_matmul::components::MatmulElems;
use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};
use cubek_std::test_utils::{
    HostData, HostDataType, HostDataVec, StrideSpec, assert_equals_approx,
};

pub fn assert_result(
    lhs: &HostData,
    rhs: &HostData,
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    dtypes: MatmulElems,
) {
    let epsilon = matmul_epsilon(&dtypes, 100.);

    let expected = matmul_cpu_reference(lhs, rhs, problem);

    let actual = HostData::from_tensor_handle(client, out, HostDataType::F32);

    if let Err(e) = assert_equals_approx(&actual, &expected, epsilon) {
        panic!("{}", e);
    }
}

fn matmul_epsilon(elems: &MatmulElems, safety_factor: f32) -> f32 {
    let total_eps = elems
        .lhs_global
        .dtype
        .epsilon()
        .max(elems.rhs_global.dtype.epsilon())
        .max(elems.acc_global.dtype.epsilon())
        .max(elems.lhs_stage.dtype.epsilon())
        .max(elems.rhs_stage.dtype.epsilon())
        .max(elems.acc_stage.dtype.epsilon())
        .max(elems.lhs_register.dtype.epsilon())
        .max(elems.rhs_register.dtype.epsilon())
        .max(elems.acc_register.dtype.epsilon());

    total_eps as f32 * safety_factor
}

/// Solves a matmul problem
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
fn matmul_cpu_reference(lhs: &HostData, rhs: &HostData, problem: &MatmulProblem) -> HostData {
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;

    let batch_shape = problem.output_batch_dims();
    let num_batches: usize = batch_shape.iter().product();
    let mut output_shape = batch_shape.clone();
    output_shape.push(m);
    output_shape.push(n);

    let mut out = vec![0.0; num_batches * m * n];

    let mut batch_index = vec![0usize; batch_shape.len()];
    let mut lhs_index = vec![0usize; batch_shape.len() + 2];
    let mut rhs_index = vec![0usize; batch_shape.len() + 2];
    let mut out_index = vec![0usize; batch_shape.len() + 2];

    // Iterate over all batches (cartesian product)
    for batch_flat in 0..num_batches {
        // decode flat batch index → multidim batch index
        let mut t = batch_flat;
        for d in (0..batch_shape.len()).rev() {
            batch_index[d] = t % batch_shape[d];
            t /= batch_shape[d];
        }

        // copy batch dims into indices
        for d in 0..batch_shape.len() {
            lhs_index[d] = batch_index[d];
            rhs_index[d] = batch_index[d];
            out_index[d] = batch_index[d];
        }

        for i in 0..m {
            out_index[batch_shape.len()] = i;
            lhs_index[batch_shape.len()] = i;

            for j in 0..n {
                out_index[batch_shape.len() + 1] = j;

                let mut sum = 0.0;
                for kk in 0..k {
                    lhs_index[batch_shape.len() + 1] = kk;
                    rhs_index[batch_shape.len()] = kk;
                    rhs_index[batch_shape.len() + 1] = j;

                    sum += lhs.get_f32(&lhs_index) * rhs.get_f32(&rhs_index);
                }

                let out_linear = batch_flat * (m * n) + i * n + j;
                out[out_linear] = sum;
            }
        }
    }

    let strides = StrideSpec::RowMajor.compute_strides(&output_shape);
    HostData {
        data: HostDataVec::F32(out),
        shape: output_shape,
        strides,
    }
}
