//! The autotune, which runs whichever setting its round elected.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubek_std::autotune::GroupPriority;
use cubek_swiglu::kernels::launch_tuned;
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, Written};

use crate::swiglu::{ELEMENTS, F32, FusedRows};

/// The autotune on the test device.
struct TunedSwiglu;

/// One case in one element.
#[derive(Debug, Clone, Copy)]
struct TunedSwigluCase {
    element: ElemType,
    rows: FusedRows,
}

impl ScopeUnderTest for TunedSwiglu {
    type Case = TunedSwigluCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let out = launch_tuned(
            case.rows.call(client, case.element),
            &cubecl::test_device(),
            GroupPriority::DEFAULT,
        );
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![case.rows.reference(case.element)]
    }

    fn tolerance(&self, case: &Self::Case) -> f32 {
        FusedRows::tolerance(case.element)
    }
}

/// Every case through the autotune.
#[test]
fn the_autotune_matches_the_reference() {
    let cases: Vec<_> = ELEMENTS
        .into_iter()
        .flat_map(|element| FusedRows::awkward().map(|rows| TunedSwigluCase { element, rows }))
        .collect();
    TunedSwiglu.check(cases);
}

/// No rows, and rows of no width: nothing to launch, and an empty output of the asked shape.
#[test]
fn empty_rows_launch_nothing() {
    let device = cubecl::test_device();
    let client = device.client();
    for rows in [
        FusedRows { rows: 0, dim: 128 },
        FusedRows { rows: 3, dim: 0 },
    ] {
        let out = launch_tuned(rows.call(&client, F32), &device, GroupPriority::DEFAULT);
        assert_eq!(out.shape().as_slice(), [rows.rows, rows.dim], "{rows:?}");
    }
}
