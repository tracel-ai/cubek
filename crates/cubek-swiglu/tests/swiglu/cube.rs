//! The cube routine at every setting it lists: the family's floor.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Decline;
use cubek_std::Hardware;
use cubek_swiglu::kernels::{CubeSwigluRoutine, SwigluRoutineCandidate};
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, Written};

use crate::swiglu::{ELEMENTS, F32, FusedRows};

/// The cube routine on the test device.
struct CubeSwiglu;

/// The routine at one of its settings, on one case, in one element.
#[derive(Clone)]
struct CubeSwigluCase {
    candidate: SwigluRoutineCandidate<CubeSwigluRoutine>,
    element: ElemType,
    rows: FusedRows,
}

impl core::fmt::Debug for CubeSwigluCase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {:?} at {:?}",
            self.candidate.settings(),
            self.rows,
            self.element
        )
    }
}

impl ScopeUnderTest for CubeSwiglu {
    type Case = CubeSwigluCase;

    /// # Panics
    ///
    /// Where the routine refuses a case: it is the floor, which serves every one.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Decline> {
        let out = case
            .candidate
            .launch(&case.rows.call(client, case.element))
            .unwrap_or_else(|why| panic!("{case:?} refused: {why:?}"));
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

/// Every setting serves every case: the routine is the family's floor.
#[test]
fn cube_routine_matches_the_reference() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw)
        .into_iter()
        .flat_map(|candidate| {
            ELEMENTS.into_iter().flat_map(move |element| {
                let candidate = candidate.clone();
                FusedRows::awkward().map(move |rows| CubeSwigluCase {
                    candidate: candidate.clone(),
                    element,
                    rows,
                })
            })
        })
        .collect();
    CubeSwiglu.check(cases);
}

/// No rows, and rows of no width: the floor launches nothing, and hands back an empty output of the
/// asked shape.
#[test]
fn empty_rows_launch_nothing() {
    let client = cubecl::test_device().client();
    let floor = SwigluRoutineCandidate::floor(&Hardware::from_client(&client));
    for rows in [
        FusedRows { rows: 0, dim: 128 },
        FusedRows { rows: 3, dim: 0 },
    ] {
        let out = floor
            .launch(&rows.call(&client, F32))
            .unwrap_or_else(|why| panic!("the floor refused {rows:?}: {why:?}"));
        assert_eq!(out.shape().as_slice(), [rows.rows, rows.dim], "{rows:?}");
    }
}
