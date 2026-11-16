//! The pooling routines against the crate's host reference: a forward case ([`Pooled`]) and a
//! backward one ([`Unpooled`]), of two or three spatial axes, and the scope each runs through. A
//! max pool that returns its indices writes them as a second output, checked exactly.

use core::ops::Range;

use cubecl::client::Client;
use cubecl::ir::{ElemType, IntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_pool::definition::{PoolBackwardProblem, PoolError, PoolForwardProblem, PoolMode};
use cubek_pool::eval::cpu_reference::{
    cpu_reference_max_pool_indices, cpu_reference_pool, cpu_reference_pool_backward,
    geometry::PoolGeometry,
};
use cubek_pool::{
    pool2d, pool2d_backward, pool2d_with_indices, pool2d_with_indices_backward, pool3d,
    pool3d_backward,
};
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

/// The element the indices of a max pool are written in.
pub(crate) const INDICES: ElemType = ElemType::Int(IntKind::I32);

/// The entry points of one rank of pooling.
pub(crate) trait PoolEntry<const N: usize> {
    /// Pool `input` into `output`, and the window's argmax into `indices` where given.
    fn forward(
        client: &Client,
        input: &TensorHandle,
        output: &TensorHandle,
        indices: Option<&TensorHandle>,
        mode: PoolMode<N>,
    ) -> Result<(), PoolError>;

    /// The input's gradient from the output's, through the argmax `indices` where given.
    fn backward(
        client: &Client,
        input: &TensorHandle,
        out_grad: &TensorHandle,
        indices: Option<&TensorHandle>,
        in_grad: &TensorHandle,
        mode: PoolMode<N>,
    ) -> Result<(), PoolError>;
}

/// The two- and three-axis entry points, chosen by the mode's rank.
pub(crate) struct Entries;

impl PoolEntry<2> for Entries {
    fn forward(
        client: &Client,
        input: &TensorHandle,
        output: &TensorHandle,
        indices: Option<&TensorHandle>,
        mode: PoolMode<2>,
    ) -> Result<(), PoolError> {
        let (input, output, dtype) = (
            input.clone().binding(),
            output.clone().binding(),
            input.dtype,
        );
        match indices {
            Some(indices) => pool2d_with_indices(
                client,
                input,
                output,
                indices.clone().binding(),
                mode,
                dtype,
            ),
            None => pool2d(client, input, output, mode, dtype),
        }
    }

    fn backward(
        client: &Client,
        input: &TensorHandle,
        out_grad: &TensorHandle,
        indices: Option<&TensorHandle>,
        in_grad: &TensorHandle,
        mode: PoolMode<2>,
    ) -> Result<(), PoolError> {
        let dtype = input.dtype;
        let (input, out_grad, in_grad) = (
            input.clone().binding(),
            out_grad.clone().binding(),
            in_grad.clone().binding(),
        );
        match indices {
            Some(indices) => pool2d_with_indices_backward(
                client,
                input,
                out_grad,
                indices.clone().binding(),
                in_grad,
                mode,
                dtype,
                indices.dtype,
            ),
            None => pool2d_backward(client, input, out_grad, in_grad, mode, dtype),
        }
    }
}

impl PoolEntry<3> for Entries {
    /// # Panics
    ///
    /// With indices, which no three-axis pool writes.
    fn forward(
        client: &Client,
        input: &TensorHandle,
        output: &TensorHandle,
        indices: Option<&TensorHandle>,
        mode: PoolMode<3>,
    ) -> Result<(), PoolError> {
        assert!(indices.is_none(), "a three-axis pool writes no indices");
        let dtype = input.dtype;
        pool3d(
            client,
            input.clone().binding(),
            output.clone().binding(),
            mode,
            dtype,
        )
    }

    /// # Panics
    ///
    /// With indices, which no three-axis pool reads.
    fn backward(
        client: &Client,
        input: &TensorHandle,
        out_grad: &TensorHandle,
        indices: Option<&TensorHandle>,
        in_grad: &TensorHandle,
        mode: PoolMode<3>,
    ) -> Result<(), PoolError> {
        assert!(indices.is_none(), "a three-axis pool reads no indices");
        let dtype = input.dtype;
        pool3d_backward(
            client,
            input.clone().binding(),
            out_grad.clone().binding(),
            in_grad.clone().binding(),
            mode,
            dtype,
        )
    }
}

/// The input of a case: `f32` values of `shape`, seeded by `seed` uniformly in `inputs`.
#[derive(Debug, Clone)]
pub(crate) struct Seeded {
    pub seed: u64,
    pub inputs: Range<f32>,
}

impl Seeded {
    /// The tensor of `shape`, and its values on the host.
    fn of(&self, shape: &[usize]) -> (TensorHandle, HostData) {
        TestInput::builder(cubecl::test_device().client(), shape.to_vec())
            .uniform(self.seed, self.inputs.start, self.inputs.end)
            .generate_with_f32_host_data()
    }

    /// The same values, seeded one further: what a second input of the case holds.
    fn next(&self) -> Self {
        Self {
            seed: self.seed + 1,
            inputs: self.inputs.clone(),
        }
    }

    /// A case's output of `shape` in `dtype`, poisoned: NaN for values, an index no window
    /// reaches for indices, so a cell the pool never writes fails the comparison.
    fn poisoned(client: &Client, shape: &[usize], dtype: ElemType) -> TensorHandle {
        let poison = match dtype {
            ElemType::Float(_) => f32::NAN,
            _ => 77_777.0,
        };
        TestInput::builder(client.clone(), shape.to_vec())
            .dtype(dtype)
            .custom(vec![poison; shape.iter().product()])
            .generate_without_host_data()
    }
}

/// One forward pool: its problem, its input, and the difference from the host it is held to.
#[derive(Debug, Clone)]
pub(crate) struct Pooled<const N: usize> {
    pub problem: PoolForwardProblem<N>,
    pub input: Seeded,
    pub tolerance: f32,
}

impl<const N: usize> Pooled<N> {
    fn output_shape(&self) -> Vec<usize> {
        self.problem
            .output_shape(&self.problem.input_shape)
            .to_vec()
    }

    /// The argmax of every window, on the host, where the max pool returns it.
    fn indices(&self, input: &HostData) -> Option<HostData> {
        match (&self.problem.mode, self.problem.with_indices) {
            (PoolMode::Max(options), true) => Some(cpu_reference_max_pool_indices(
                input,
                options,
                &self.output_shape(),
            )),
            _ => None,
        }
    }
}

/// The forward pool of `N` spatial axes, against the host: the output, and the indices where the
/// case asks for them.
pub(crate) struct PoolForward<const N: usize>;

impl<const N: usize> ScopeUnderTest for PoolForward<N>
where
    Entries: PoolEntry<N>,
{
    type Case = Pooled<N>;

    fn launch(&self, client: &Client, case: &Pooled<N>) -> Result<Vec<Written>, Refusal> {
        let (input, _) = case.input.of(&case.problem.input_shape);
        let shape = case.output_shape();
        let output = Seeded::poisoned(client, &shape, input.dtype);
        let indices = case
            .problem
            .with_indices
            .then(|| Seeded::poisoned(client, &shape, INDICES));
        Entries::forward(
            client,
            &input,
            &output,
            indices.as_ref(),
            case.problem.mode.clone(),
        )
        .unwrap_or_else(|why| panic!("{case:?} refused: {why:?}"));
        let mut written = vec![Written {
            name: "output",
            tensor: output,
            host: HostDataType::F32,
        }];
        written.extend(indices.map(|indices| Written {
            name: "indices",
            tensor: indices,
            host: HostDataType::I32,
        }));
        Ok(written)
    }

    fn reference(&self, case: &Pooled<N>) -> Vec<HostData> {
        let (_, input) = case.input.of(&case.problem.input_shape);
        let mut reference = vec![cpu_reference_pool(&input, case.problem.clone())];
        reference.extend(case.indices(&input));
        reference
    }

    fn tolerance(&self, case: &Pooled<N>) -> f32 {
        case.tolerance
    }

    /// The argmax a max pool writes is exact.
    fn tolerance_of(&self, case: &Pooled<N>, name: &'static str) -> f32 {
        match name {
            "indices" => 0.,
            _ => case.tolerance,
        }
    }
}

/// One backward pool: its problem, its input and output gradient, and the difference from the
/// host it is held to. A max pool with indices first pools the input forward for its argmax.
#[derive(Debug, Clone)]
pub(crate) struct Unpooled<const N: usize> {
    pub problem: PoolBackwardProblem<N>,
    pub input: Seeded,
    pub tolerance: f32,
}

impl<const N: usize> Unpooled<N> {
    fn input_shape(&self) -> Vec<usize> {
        let grad = &self.problem.out_grad_shape;
        [
            vec![grad[0]],
            self.problem.input_size.to_vec(),
            vec![grad[N + 1]],
        ]
        .concat()
    }

    /// The argmax of every window, on the host, where the max pool reads it.
    fn indices(&self, input: &HostData) -> Option<HostData> {
        match (&self.problem.mode, self.problem.with_indices) {
            (PoolMode::Max(options), true) => Some(cpu_reference_max_pool_indices(
                input,
                options,
                &self.problem.out_grad_shape,
            )),
            _ => None,
        }
    }
}

/// The backward pool of `N` spatial axes, against the host: the input's gradient, and the indices
/// the forward pool wrote for it where the case reads them.
pub(crate) struct PoolBackward<const N: usize>;

impl<const N: usize> ScopeUnderTest for PoolBackward<N>
where
    Entries: PoolEntry<N>,
{
    type Case = Unpooled<N>;

    fn launch(&self, client: &Client, case: &Unpooled<N>) -> Result<Vec<Written>, Refusal> {
        let input_shape = case.input_shape();
        let grad_shape = case.problem.out_grad_shape.to_vec();
        let (input, _) = case.input.of(&input_shape);
        let (out_grad, _) = case.input.next().of(&grad_shape);
        let in_grad = Seeded::poisoned(client, &input_shape, input.dtype);
        let mode = case.problem.mode.clone();
        let indices = case.problem.with_indices.then(|| {
            let indices = Seeded::poisoned(client, &grad_shape, INDICES);
            let pooled = Seeded::poisoned(client, &grad_shape, input.dtype);
            Entries::forward(client, &input, &pooled, Some(&indices), mode.clone())
                .unwrap_or_else(|why| panic!("{case:?}: the forward pool refused: {why:?}"));
            indices
        });
        Entries::backward(client, &input, &out_grad, indices.as_ref(), &in_grad, mode)
            .unwrap_or_else(|why| panic!("{case:?} refused: {why:?}"));
        let mut written = vec![Written {
            name: "input gradient",
            tensor: in_grad,
            host: HostDataType::F32,
        }];
        written.extend(indices.map(|indices| Written {
            name: "indices",
            tensor: indices,
            host: HostDataType::I32,
        }));
        Ok(written)
    }

    fn reference(&self, case: &Unpooled<N>) -> Vec<HostData> {
        let (_, input) = case.input.of(&case.input_shape());
        let (_, out_grad) = case.input.next().of(&case.problem.out_grad_shape);
        let indices = case.indices(&input);
        // The reference reads the argmax for a max pool, and nothing it reads for the others.
        let read = indices.as_ref().unwrap_or(&input);
        let mut reference = vec![cpu_reference_pool_backward(
            &out_grad,
            read,
            case.problem.clone(),
        )];
        reference.extend(indices);
        reference
    }

    fn tolerance(&self, case: &Unpooled<N>) -> f32 {
        case.tolerance
    }

    /// The argmax a max pool writes is exact.
    fn tolerance_of(&self, case: &Unpooled<N>, name: &'static str) -> f32 {
        match name {
            "indices" => 0.,
            _ => case.tolerance,
        }
    }
}
