//! What the reduce routines' Small Tests share: a routine launched by a function of the test's,
//! below the strategies the Operation offers, on data whose every sum is exact, against the host.

use crate::eval::cpu_reference;
use crate::kernels::{ReduceElements, ReduceLauncher, ReduceTensors};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Decline, StoredTileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubek_microkernel::reduce::{ReduceOperation, ReduceWrites};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// Small whole numbers, so every sum is exact, with ties and zeros; for a product, `±1`, `2` and
/// `0.5` on a cycle of seven, prime to every extent here so no row repeats one value, with as many
/// `2` as `0.5` so that a long row's product stays exact and finite. A NaN, an infinity or a row
/// of ties is a behavior of its own, which the microkernels' Component Tests check.
fn values(len: usize, product: bool) -> Vec<f32> {
    (0..len)
        .map(|i| match product {
            true => [1.0, -1.0, 2.0, 0.5, 1.0, 2.0, 0.5][i % 7],
            false => whole_value(i),
        })
        .collect()
}

/// Value `i` of a row that is not a product's: a run of ties, a zero, and between them small whole
/// numbers on a cycle of eleven.
fn whole_value(i: usize) -> f32 {
    match i % 97 {
        29 | 30 => 9.0,
        50 => 0.0,
        _ => ((i * 7) % 11) as f32 - 5.0,
    }
}

/// `shape` reduced along `axis` by `operation`.
#[derive(Debug, Clone)]
pub(crate) struct Reduction {
    /// The input's extent along each axis, read by axis (Position Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
}

impl Reduction {
    fn input(&self) -> Vec<f32> {
        values(
            self.shape.iter().product(),
            self.operation == ReduceOperation::Prod,
        )
    }

    fn out_elem(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }

    fn out_shape(&self) -> Vec<usize> {
        let mut out_shape = self.shape.clone();
        out_shape[self.axis] = self.operation.written_per_row();
        out_shape
    }

    fn elements(&self) -> ReduceElements {
        ReduceElements {
            input: F32,
            accumulator: F32,
            out: self.out_elem(),
        }
    }

    /// The tensors this reduction reads from `input` and writes to `out`, bound.
    pub fn tensors(
        &self,
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
    ) -> ReduceTensors {
        ReduceTensors::new(
            client,
            input,
            out,
            self.axis,
            self.operation,
            self.elements(),
        )
    }
}

/// How a test launches its routine on a reduction: from `input` into `out`, or the routine's
/// decline.
pub(crate) type Launch =
    fn(&Client, &Reduction, TensorBinding, TensorBinding) -> Result<(), Decline>;

/// A reduce routine, launched as the test's [`Launch`] launches it, against the host. Its output
/// starts away from zero, so a routine that zeroes it first is checked to.
pub(crate) struct RoutineScopeUnderTest(pub Launch);

impl ScopeUnderTest for RoutineScopeUnderTest {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let (out_elem, out_shape) = (reduction.out_elem(), reduction.out_shape());
        let input = TestInput::builder(client.clone(), Shape::from(reduction.shape.clone()))
            .dtype(F32)
            .custom(reduction.input())
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), Shape::from(out_shape.clone()))
            .dtype(out_elem)
            .custom(vec![-7.0; out_shape.iter().product()])
            .generate_without_host_data();
        (self.0)(client, reduction, input.binding(), out.clone().binding())?;
        let host = match out_elem {
            I32 => HostDataType::I32,
            _ => HostDataType::F32,
        };
        Ok(vec![Written {
            name: "output",
            tensor: out,
            host,
        }])
    }

    /// What the reduce reference writes for the input.
    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let input = HostData::row_major(
            HostDataVec::F32(reduction.input()),
            Shape::from(reduction.shape.clone()),
        );
        vec![cpu_reference::reference(
            &input,
            reduction.axis,
            reduction.operation,
            None,
        )]
    }

    /// Whole numbers sum exactly; a mean divides once.
    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        1e-6
    }
}

/// The tensors `R` reduces unbound: `input`'s geometry, stated at the width the routine picks
/// for it bound, and `out`.
pub(crate) fn unbound_tensors<R: ReduceLauncher<M>, M>(
    client: &Client,
    input: &TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> Result<ReduceTensors<Geometry>, Decline> {
    let bound = reduction.tensors(client, input.clone(), out.clone());
    let hw = Hardware::from_client(client);
    let (blueprint, _) = R::prepare_first_served(&hw, bound.problem())?;
    Ok(ReduceTensors::unbound(
        client,
        Geometry::from(input),
        blueprint.width,
        out,
        reduction.axis,
        reduction.operation,
        reduction.elements(),
    ))
}

/// `input` bound back as the plain buffer it is, at the geometry, spec and width the launch
/// derived: its row axes may be merged into one.
pub(crate) fn bound_back(mut input: TensorBinding) -> impl FnOnce(Unbound) -> TileTensorLaunch {
    move |bound| {
        input.shape = bound.geometry.shape().iter().copied().collect();
        input.strides = bound.geometry.strides().iter().copied().collect();
        TileTensorLaunch::new(input.into_tensor_arg(), F32, bound.vector_size, bound.spec)
    }
}

/// `R` launched unbound over `input`, as a fused read stands in for, bound back as the buffer it
/// is, into `out`.
pub(crate) fn launch_unbound<R: ReduceLauncher<M>, M>(
    client: &Client,
    reduction: &Reduction,
    input: TensorBinding,
    out: TensorBinding,
) -> Result<(), Decline> {
    let tensors = unbound_tensors::<R, M>(client, &input, out, reduction)?;
    let hw = Hardware::from_client(client);
    let (blueprint, grid) = R::prepare_first_served(&hw, tensors.problem())?;
    R::launch_unbound::<StoredTileOperand>(&tensors, blueprint, grid, bound_back(input))
}

/// A test for each `name: operation`, each one launch by `launch` of `shape` reduced along `axis`,
/// checked against the host; a routine that declines it is a skip.
macro_rules! reduction_tests {
    (
        $launch:expr,
        $shape:expr,
        $axis:expr,
        [$($name:ident: $operation:expr),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                use cubek_test_utils::ScopeUnderTest;
                use $crate::kernels::routines::routine_check::{Reduction, RoutineScopeUnderTest};
                RoutineScopeUnderTest($launch).check([Reduction {
                    shape: $shape.to_vec(),
                    axis: $axis,
                    operation: $operation,
                }]);
            }
        )*
    };
}
pub(crate) use reduction_tests;

/// `routine` launched unbound, the input bound back as the plain buffer it is: rows 300 long along
/// the contiguous dim, and rows 5 long across it.
macro_rules! reduces_unbound_each_dim {
    ($routine:ty) => {
        /// Launched unbound, as a fused read stands in for.
        mod unbound {
            /// Rows 300 long, along the contiguous dim.
            mod along_the_contiguous_dim {
                use $crate::ReduceOperation;
                use $crate::kernels::routines::routine_check::{launch_unbound, reduction_tests};
                reduction_tests!(launch_unbound::<$routine, _>, [2, 5, 300], 2, [
                    sum: ReduceOperation::Sum,
                    max: ReduceOperation::Max,
                    mean: ReduceOperation::Mean,
                    arg_max: ReduceOperation::ArgMax,
                ]);
            }

            /// Rows 5 long, across the contiguous dim.
            mod across_the_middle_dim {
                use $crate::ReduceOperation;
                use $crate::kernels::routines::routine_check::{launch_unbound, reduction_tests};
                reduction_tests!(launch_unbound::<$routine, _>, [2, 5, 300], 1, [
                    sum: ReduceOperation::Sum,
                    max: ReduceOperation::Max,
                    mean: ReduceOperation::Mean,
                    arg_max: ReduceOperation::ArgMax,
                ]);
            }
        }
    };
}
pub(crate) use reduces_unbound_each_dim;
