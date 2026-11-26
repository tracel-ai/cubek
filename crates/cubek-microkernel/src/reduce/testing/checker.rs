use crate::reduce::ReduceWorkItem;
use crate::reduce::testing::{
    F32, HostReduce, LaunchedRows, MicrokernelLauncher, ReduceCase, RowShare,
};
use cubecl::client::Client;
use cubecl::tile::launch::Decline;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, ScopeChecker, TestInput, Written};

/// Checks the reduce microkernel `M` launches against the reduce reference; a microkernel the device
/// does not run is a skip.
pub(crate) struct ReduceMicrokernelScopeChecker<M> {
    hw: Hardware,
    microkernel: M,
}

impl<M: MicrokernelLauncher> ReduceMicrokernelScopeChecker<M> {
    /// The checker of `microkernel` on the test device.
    pub(crate) fn new(microkernel: M) -> Self {
        Self {
            hw: Hardware::from_client(&cubecl::test_device().client()),
            microkernel,
        }
    }

    /// `case` on the host.
    fn host(&self, case: &ReduceCase) -> HostReduce {
        let share = RowShare {
            part_count: self.microkernel.parts(),
            stride_count: self.microkernel.stride_on(&self.hw),
            value_count: self.microkernel.values_of_value_count(case.value_count),
        };
        HostReduce::new(*case, share)
    }
}

impl<M: MicrokernelLauncher> ScopeChecker for ReduceMicrokernelScopeChecker<M> {
    type Case = ReduceCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Decline> {
        let host = self.host(case);
        self.microkernel.serves_out(&self.hw, case.out())?;
        let item = ReduceWorkItem {
            operation: case.operation,
            input: F32,
            accumulator: F32,
            lines_along: case.lines.to_lines_along(),
            with_indices: false,
        };
        let rows = LaunchedRows {
            item,
            row_count: host.rows.len(),
            value_count: case.value_count,
            line_lane_count: case.lane_count,
            plane_lane_count: self.microkernel.plane_width(&self.hw, &item)?,
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
            .custom(vec![self.microkernel.unwritten(); cells])
            .generate_without_host_data();
        self.microkernel.launch(
            client,
            input.binding().into_tensor_arg(),
            out.clone().binding(),
            rows,
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![self.host(case).out()]
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-4
    }
}
