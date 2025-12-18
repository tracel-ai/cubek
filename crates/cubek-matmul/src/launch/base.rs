use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl::std::tensor::TensorHandle;

use crate::launch::handle::{MatmulInputHandle, MatmulInputHandleRef};
use crate::{
    definition::{MatmulElems, MatmulSetupError},
    launch::Strategy,
};

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: MatmulInputHandle<R>,
    rhs: MatmulInputHandle<R>,
    out: TensorHandle<R>,
    mut dtypes: MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_ref(
        strategy,
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
        &mut dtypes,
    )
}

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
    lhs: &MatmulInputHandleRef<R>,
    rhs: &MatmulInputHandleRef<R>,
    out: &TensorHandleRef<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    strategy.launch_ref(client, lhs, rhs, out, dtypes)
}
