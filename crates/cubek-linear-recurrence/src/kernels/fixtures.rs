//! Devices the routine tests prepare for.

use cubek_std::Hardware;
use cubek_std::devices;

/// `hw` as a device that binds as many buffers as a launch of the family does: the fixtures that
/// bind eight decline the launches that bind more, and the routine tests would otherwise prepare
/// them on none of them.
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
