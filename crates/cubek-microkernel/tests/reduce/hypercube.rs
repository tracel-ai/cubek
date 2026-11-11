//! The hypercube reduce microkernel alone: cubes of four planes each reduce a part of the row, the
//! last to finish merging the parts.

use crate::launch::reduce::ReduceScope;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::reduce::{Reduce, ReduceCase};

/// A row split across cubes, four planes each, reduces every operation to the CPU's answer: along
/// the reduced axis, over two parts and over eight, a part shorter than the cube's units and one
/// several times longer; and across it over three, each lane its own row.
#[test]
fn a_hypercube_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let hypercube = |parts| ReduceScope::Hypercube { planes: 4, parts };
    let mut cases = ReduceCase::every_operation(hypercube(2), &[(64, 1)], &[]);
    cases.extend(ReduceCase::every_operation(hypercube(8), &[(8192, 4)], &[]));
    cases.extend(ReduceCase::every_operation(hypercube(3), &[], &[(300, 4)]));
    Reduce { hw }.check(cases);
}
