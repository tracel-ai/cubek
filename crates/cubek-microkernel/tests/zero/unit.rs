//! The unit zero microkernel alone: a plane's first unit zeroes its row of an output that held
//! something else.

use crate::launch::zero::launch_unit_zero;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::launch::Decline;
use cubecl::zspace::Shape;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

/// An output of `rows` rows of `cells` cells in `element`, holding sevens before the launch.
#[derive(Debug, Clone, Copy)]
struct ZeroCase {
    pub rows: usize,
    pub cells: usize,
    pub element: ElemType,
}

struct UnitZeroScopeUnderTest {
    hw: Hardware,
}

impl ScopeUnderTest for UnitZeroScopeUnderTest {
    type Case = ZeroCase;

    fn launch(&self, client: &Client, case: &ZeroCase) -> Result<Vec<Written>, Decline> {
        let shape = Shape::from(vec![case.rows, case.cells]);
        let out = TestInput::builder(client.clone(), shape)
            .dtype(case.element)
            .custom(vec![7.0; case.rows * case.cells])
            .generate_without_host_data();
        let plane_width = self.hw.plane_dim().unwrap_or(1);
        launch_unit_zero(
            client,
            out.clone().binding().into_tensor_arg(),
            case.element,
            case.rows,
            case.cells,
            plane_width,
        )?;
        let host = match case.element {
            ElemType::Int(_) => HostDataType::I32,
            _ => HostDataType::F32,
        };
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host,
        }])
    }

    fn reference(&self, case: &ZeroCase) -> Vec<HostData> {
        vec![HostData::row_major(
            HostDataVec::F32(vec![0.0; case.rows * case.cells]),
            Shape::from(vec![case.rows, case.cells]),
        )]
    }

    fn tolerance(&self, _case: &ZeroCase) -> f32 {
        0.0
    }
}

/// A unit alone zeroes its row in `element`, at a row length that is not a power of two.
fn a_unit_zeroes_its_row(element: ElemType) {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    UnitZeroScopeUnderTest { hw }.check([ZeroCase {
        rows: 5,
        cells: 37,
        element,
    }]);
}

#[test]
fn a_unit_zeroes_a_row_of_floats() {
    a_unit_zeroes_its_row(ElemType::Float(FloatKind::F32));
}

#[test]
fn a_unit_zeroes_a_row_of_integers() {
    a_unit_zeroes_its_row(ElemType::Int(IntKind::I32));
}
