//! The unit zero microkernel alone: a plane's first unit zeroes its row of an output that held
//! something else.

use crate::launch::zero::launch_unit_zero;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

/// An output of `rows` rows of `cells` cells in `element`, holding sevens before the launch.
#[derive(Debug, Clone, Copy)]
struct ZeroCase {
    rows: usize,
    cells: usize,
    element: ElemType,
}

struct UnitZeroScopeUnderTest {
    hw: Hardware,
}

impl ScopeUnderTest for UnitZeroScopeUnderTest {
    type Case = ZeroCase;

    fn launch(&self, client: &Client, case: &ZeroCase) -> Result<Vec<Written>, Refusal> {
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

/// A unit alone zeroes its row, in a float and an integer, at a row length that is not a power of
/// two.
#[test]
fn a_unit_zeroes_its_row() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases =
        [ElemType::Float(FloatKind::F32), ElemType::Int(IntKind::I32)].map(|element| ZeroCase {
            rows: 5,
            cells: 37,
            element,
        });
    UnitZeroScopeUnderTest { hw }.check(cases);
}
