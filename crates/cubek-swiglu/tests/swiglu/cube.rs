//! The cube routine at every setting it lists: its own autotune's floor.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_swiglu::kernels::{CubeSwigluRoutine, SwigluRoutineCandidate};
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, Written};

use crate::swiglu::{ELEMENTS, FusedRows};

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
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
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

/// Every setting serves every case: the routine is its own autotune's floor.
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
