use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::launch::handle::MatmulInputBinding;
use crate::{
    definition::{MatmulElems, MatmulSetupError},
    launch::Strategy,
};

#[allow(clippy::result_large_err)]
/// Launches a matrix multiplication kernel..
///
/// # Notes
///
/// The matmul elements may get changed during selection for improved performance when
/// the hardware supports it.
/// Only the inner element types may change such as the stage or register element types.
pub fn launch_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: MatmulInputBinding<R>,
    rhs: MatmulInputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    strategy.launch_ref(client, lhs, rhs, out, dtypes)
}
