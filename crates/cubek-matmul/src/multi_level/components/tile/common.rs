//! Shared helpers used by per-variant validators.

use cubecl::{
    features::{Plane as PlaneFeature, TypeUsage},
    ir::{ElemType, FloatKind},
    prelude::*,
};

use crate::definition::{MatmulAvailabilityError, MatmulElems, MatmulSetupError};

pub(super) fn check_types_available<R: Runtime>(
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
    require_plane_ops: bool,
) -> Result<(), MatmulSetupError> {
    if require_plane_ops
        && !client
            .properties()
            .features
            .plane
            .contains(PlaneFeature::Ops)
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneOpsUnavailable,
        ));
    }

    let lhs = normalize_flex32(dtypes.lhs_register);
    let rhs = normalize_flex32(dtypes.rhs_register);
    let output = normalize_flex32(dtypes.acc_register);

    if !(client
        .properties()
        .features
        .type_usage(lhs)
        .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(rhs)
            .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(output)
            .contains(TypeUsage::Arithmetic))
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output },
        ));
    }

    Ok(())
}

fn normalize_flex32(ty: ElemType) -> ElemType {
    match ty {
        ElemType::Float(FloatKind::Flex32) => ElemType::Float(FloatKind::F32),
        _ => ty,
    }
}
