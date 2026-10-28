use super::base::{
    MatmulProduct, MatmulTableGroups, MatmulTunables, RoutineIdentity, measured_table,
    reads_no_routes,
};
use crate::kernels::MatmulRoutine;
use crate::kernels::plan::MatmulCall;
use crate::kernels::tune::key::MatmulTuneKey;
use crate::kernels::{CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulTensors};
use cubecl::tune::Tunable;
use cubek_std::Hardware;
use cubek_std::autotune::{GroupPriority, Identity};
use cubek_std::routine::Routine;

/// The NVFP4 table's candidates on `hw`: the staged routine reading both inputs as stored on the
/// 4-bit tensor cores, at each of its block-scaled settings
/// ([`block_scaled_settings`](CubeStagedMatmulRoutine::block_scaled_settings)), the last, its
/// default, on the floor, since it serves every product the selector sends here.
///
/// Every candidate sums each output in the same order, so the race moves the time and no number.
/// Not bounded: a roofline is a 16-bit product's peak, which the 4-bit instruction runs past, so a
/// bound would end every round at its first candidate. Its one group is its own, whatever the
/// caller's groups say: the table races one kernel's settings.
pub(super) fn table(hw: Hardware, _groups: MatmulTableGroups) -> MatmulTunables {
    let mut table = measured_table();
    let group = table.group("nvfp4", GroupPriority::DEFAULT);
    let settings = CubeStagedMatmulRoutine.block_scaled_settings(&hw);
    let (&floor, raced) = settings
        .split_last()
        .expect("the NVFP4 settings end in the default");
    let table = raced.iter().fold(table, |table, &settings| {
        table.candidate(&group, candidate(hw.clone(), settings))
    });
    table.floor(candidate(hw, floor)).finish()
}

/// `settings` as a candidate of the NVFP4 table: both inputs contracted as they are stored.
fn candidate(
    hw: Hardware,
    settings: CubeStagedMatmulRoutineSettings,
) -> Tunable<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity> {
    RoutineIdentity::named(&format!("nvfp4 {settings}"), move |call: MatmulCall| {
        reads_no_routes(&call)?;
        let tensors =
            MatmulTensors::written_in(&call.client, call.lhs.tensor, call.rhs.tensor, call.out)
                .map_err(|why| format!("{settings} does not serve this product: {why:?}"))?;
        let (blueprint, grid) = CubeStagedMatmulRoutine
            .prepare(&hw, &tensors.problem(), &settings)
            .map_err(|why| format!("{settings} does not serve this product: {why:?}"))?;
        CubeStagedMatmulRoutine.dispatch_into_buffer(&tensors, blueprint, grid);
        Ok::<_, String>(MatmulProduct {
            client: call.client,
            handle: tensors
                .into_out()
                .expect("a launch allocating its output hands it back"),
        })
    })
}
