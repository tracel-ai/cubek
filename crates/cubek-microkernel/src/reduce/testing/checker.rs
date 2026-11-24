use crate::reduce::ReduceWorkItem;
use crate::reduce::testing::{F32, HostReduce, LaunchedRows, MicrokernelLauncher, ReduceCase};
use cubecl::client::Client;
use cubecl::tile::launch::Decline;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, ScopeChecker, TestInput, Written};

/// Checks the reduce microkernel `M` launches against the reduce reference.
struct ReduceMicrokernelScopeChecker<M> {
    hw: Hardware,
    microkernel: core::marker::PhantomData<M>,
}

/// `case` launched once and checked against the reduce reference; a microkernel the device does
/// not run is a skip.
pub(crate) fn reduces_like_the_cpu<M: MicrokernelLauncher>(case: ReduceCase<M>) {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let microkernel = core::marker::PhantomData;
    ReduceMicrokernelScopeChecker { hw, microkernel }.check([case]);
}

impl<M: MicrokernelLauncher> ScopeChecker for ReduceMicrokernelScopeChecker<M> {
    type Case = ReduceCase<M>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Decline> {
        let host = HostReduce::new(*case, &self.hw);
        let item = ReduceWorkItem {
            operation: case.operation,
            input: F32,
            accumulator: F32,
            lines_along: case.lines.to_lines_along(),
            with_indices: false,
        };
        let rows = LaunchedRows {
            item,
            rows: host.rows.len(),
            reduced: case.len,
            width: case.width,
            lanes: case.microkernel.plane_width(&self.hw, &item)?,
            out: case.out(),
        };
        let (input_shape, input_values) = host.input();
        let out_shape = host.out_shape();
        let cells = out_shape.num_elements();
        let input = TestInput::builder(client.clone(), input_shape)
            .dtype(F32)
            .custom(input_values)
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), out_shape)
            .dtype(case.out())
            .custom(vec![-7.0; cells])
            .generate_without_host_data();
        case.microkernel.launch(
            client,
            input.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
            rows,
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![HostReduce::new(*case, &self.hw).out()]
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-4
    }
}
