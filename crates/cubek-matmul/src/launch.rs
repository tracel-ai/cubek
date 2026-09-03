use cubecl::{client::Client, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::{
    definition::{MatmulElems, MatmulSetupError},
    strategy::Strategy,
};

#[allow(clippy::result_large_err)]
/// Launches a matrix multiplication kernel..
///
/// # Notes
///
/// The matmul elements may get changed during selection for improved performance when
/// the hardware supports it.
/// Only the inner element types may change such as the stage or register element types.
pub fn launch_ref(
    strategy: &Strategy,
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    strategy.launch_ref(client, lhs, rhs, out, dtypes)
}
