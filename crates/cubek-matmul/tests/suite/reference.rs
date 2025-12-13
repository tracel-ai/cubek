use cubecl::TestRuntime;
use cubecl::std::tensor::TensorHandle;
use cubecl::{CubeElement, client::ComputeClient};
use cubek_matmul::components::MatmulElems;
use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};
use cubek_std::test_utils::{HostData, HostDataType, HostDataVec, assert_equals_approx};

pub fn assert_result(
    lhs: &HostData,
    rhs: &HostData,
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    dtypes: MatmulElems,
) {
    let epsilon = matmul_epsilon(&dtypes, 100.);

    println!("Executing CPU reference");
    let expected = matmul_cpu_reference(lhs, rhs, problem);

    println!("Turning output into host data");
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

// fn matmul_cpu_reference(lhs: &[f32], rhs: &[f32], problem: &MatmulProblem) -> Vec<f32> where {
//     let m = problem.m;
//     let n = problem.n;
//     let k = problem.k;
//     let num_batches = problem.num_batches();
//     let b_lhs = problem.lhs_batches.clone();
//     let b_rhs = problem.rhs_batches.clone();
//     assert!(
//         b_lhs.len() == b_rhs.len(),
//         "Cpu reference only works with batches of equal length. Please pad the shortest one with ones at the beginning."
//     );
//     let lhs_strides = strides(problem, MatmulIdent::Lhs);
//     let rhs_strides = strides(problem, MatmulIdent::Rhs);
//     let out_strides = strides(problem, MatmulIdent::Out);
//     let mut acc = vec![0.; m * n * num_batches];
//     for nth_batch in 0..num_batches {
//         let batch_out = nth_batch * m * n;
//         let mut batch_lhs = 0;
//         let mut batch_rhs = 0;
//         for b in 0..b_lhs.len() {
//             let tmp = batch_out / out_strides[b];
//             batch_lhs += tmp % b_lhs[b] * lhs_strides[b];
//             batch_rhs += tmp % b_rhs[b] * rhs_strides[b];
//         }
//         for i in 0..m {
//             for j in 0..n {
//                 for k_ in 0..k {
//                     let lhs_index = i * k + k_;
//                     let rhs_index = k_ * n + j;
//                     let out_index = i * n + j;
//                     let l = lhs[batch_lhs + lhs_index];
//                     let r = rhs[batch_rhs + rhs_index];
//                     let prod = l * r;
//                     acc[batch_out + out_index] += prod;
//                 }
//             }
//         }
//     }
//     acc
// }

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

                    sum += lhs.get(&lhs_index) * rhs.get(&rhs_index);
                }

                let out_linear = batch_flat * (m * n) + i * n + j;
                out[out_linear] = sum;
            }
        }
    }

    let strides = row_major_strides(&output_shape);
    HostData {
        data: HostDataVec::F32(out),
        shape: output_shape,
        strides,
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

/// Returns the stride of the identified tensor, inferred by the problem definition
fn strides(problem: &MatmulProblem, ident: MatmulIdent) -> Vec<usize> {
    let shape = problem.shape(ident);
    let rank = shape.len();
    let mut strides = Vec::with_capacity(rank);

    let (last_batch, x, y) = match ident {
        MatmulIdent::Lhs => match problem.lhs_layout {
            MatrixLayout::RowMajor => (problem.m * problem.k, problem.k, 1),
            MatrixLayout::ColMajor => (problem.m * problem.k, 1, problem.m),
        },
        MatmulIdent::Rhs => match problem.rhs_layout {
            MatrixLayout::RowMajor => (problem.k * problem.n, problem.n, 1),
            MatrixLayout::ColMajor => (problem.k * problem.n, 1, problem.k),
        },
        MatmulIdent::Out => (problem.m * problem.n, problem.n, 1),
    };

    strides.push(y);
    strides.push(x);

    if rank > 2 {
        strides.push(last_batch);

        for b in shape.iter().rev().take(rank - 3) {
            strides.push(last_batch * b)
        }
    }

    strides.into_iter().rev().collect()
}
