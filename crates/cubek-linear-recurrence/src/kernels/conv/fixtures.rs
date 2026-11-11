//! Devices the conv tests prepare for.

use cubek_std::Hardware;
use cubek_std::devices;

/// `hw` as a device that binds as many buffers as the conv's launch does.
pub(crate) fn binding_enough(mut hw: Hardware) -> Hardware {
    devices::topology(&mut hw).max_bindings = 32;
    hw
}

pub(crate) fn fleet() -> [Hardware; 6] {
    [
        devices::h100(),
        devices::cdna_wave64(),
        devices::cpu_12(),
        devices::strix_halo(),
        devices::rdna_range(),
        devices::wgpu_no_subgroups(),
    ]
    .map(binding_enough)
}
