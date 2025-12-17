use cubecl::TestRuntime;
use cubecl::std::tensor::TensorHandle;
use cubecl::{CubeElement, client::ComputeClient};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::definition::{MatmulIdent, MatmulProblem, MatrixLayout};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, StrideSpec, assert_equals_approx};

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

    let out_shape = problem.out_shape.clone();
    let rank = out_shape.len();
    let num_batches = problem.num_batches();

    let mut out = vec![0.0; num_batches * m * n];

    let mut batch_index = vec![0usize; rank - 2];
    let mut lhs_index = vec![0usize; rank];
    let mut rhs_index = vec![0usize; rank];
    let mut out_index = vec![0usize; rank];

    // Iterate over all batches (cartesian product)
    for batch_flat in 0..num_batches {
        // decode flat batch index → multidim batch index
        let mut t = batch_flat;
        for d in (0..rank - 2).rev() {
            batch_index[d] = t % out_shape[d];
            t /= out_shape[d];
        }

        // copy batch dims into indices
        lhs_index[..rank - 2].copy_from_slice(&batch_index);
        rhs_index[..rank - 2].copy_from_slice(&batch_index);
        out_index[..rank - 2].copy_from_slice(&batch_index);

        for i in 0..m {
            out_index[rank - 2] = i;
            lhs_index[rank - 2] = i;

            for j in 0..n {
                out_index[rank - 1] = j;

                let mut sum = 0.0;
                for kk in 0..k {
                    lhs_index[rank - 1] = kk;
                    rhs_index[rank - 2] = kk;
                    rhs_index[rank - 1] = j;

                    sum += lhs.get_f32(&lhs_index) * rhs.get_f32(&rhs_index);
                }

                let out_linear = batch_flat * (m * n) + i * n + j;
                out[out_linear] = sum;
            }
        }
    }

    let strides = StrideSpec::RowMajor.compute_strides(&out_shape);
    HostData {
        data: HostDataVec::F32(out),
        shape: out_shape,
        strides,
    }
}
