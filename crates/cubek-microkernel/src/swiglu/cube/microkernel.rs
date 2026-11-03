use super::base::CubeSwigluMicrokernel;
use crate::row_store::{CubeStage, CubeStore};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's runs, a group of `microkernel.units` units each: `out = silu(gate) ⊙ up`, each unit
/// of a group its lines of the group's run. `input` is windowed to the cube's box, both halves of
/// it: the gate's run, then the up's; `out` is the cube's window of the output, which its groups
/// write through one store, straight or staged and encoded as `stage` says.
///
/// A unit's lines are `unit + i · units`; where the units do not divide the run, a unit's last
/// one lies past it and is neither read nor written.
#[cube]
pub fn cube_swiglu_microkernel<E: Float, N: Size>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    #[comptime] microkernel: CubeSwigluMicrokernel,
    #[comptime] stage: CubeStage,
) {
    let mut store = CubeStore::new::<E, N>(out.clone(), stage);
    let (units, run_lines) = comptime!((microkernel.units, microkernel.run_lines));
    let unrolled = comptime!(microkernel.unrolled());
    for group in cube {
        // A group's units lie along `x` and the cube's groups along `y`, as its resources state
        // them.
        let (slot, unit) = (UNIT_POS_Y as usize, UNIT_POS_X as usize);
        let run_in = input.at(&group);
        let halves = run_in.flat::<E, N>();
        let mut run_out = store.owner(&group, slot * run_lines);
        #[unroll(unrolled)]
        for i in 0..comptime!(microkernel.lines_per_unit()) {
            let line = unit + i * units;
            if comptime!(run_lines.is_multiple_of(units)) || line < run_lines {
                let (gate, up) = (halves.read(line), halves.read(line + run_lines));
                run_out.write::<E, N>(line, gated::<E, N>(gate, up));
            }
        }
    }
    store.finish();
}

/// `g · sigmoid(g) · u` per element of the line, in `f32`. The elements are walked explicitly, as a
/// packed product's unpack does, so only scalar float intrinsics are needed.
#[cube]
fn gated<E: Float, N: Size>(gate: Vector<E, N>, up: Vector<E, N>) -> Vector<E, N> {
    let mut out = Vector::<E, N>::empty();
    #[unroll]
    for element in 0..N::value() {
        let g = f32::cast_from(gate.extract(element));
        let u = f32::cast_from(up.extract(element));
        let silu = g / (1.0 + (-g).exp());
        out.insert(element, E::cast_from(silu * u));
    }
    out
}
