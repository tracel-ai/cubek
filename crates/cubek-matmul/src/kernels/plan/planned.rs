use super::{Decoded, FrameworkDequantized, MatmulCall, RoutinePlan};
use crate::kernels::select::selector::{
    MatmulChoice, MatmulDequantize, MatmulPackedChoice, MatmulSelection, MatmulTable,
};
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulKernel,
    PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings,
};
use cubecl::std::tensor::TensorHandle;
use cubek_std::plan::{BoxedPlan, InOrder, Then};

/// A plan of the matmul family: what a product's launch tries, on one [`MatmulCall`].
pub type MatmulPlan = BoxedPlan<MatmulCall, TensorHandle>;

/// The plan that runs `selection`, the tables it names reached through `table`.
pub fn planned(
    selection: &MatmulSelection,
    table: &dyn Fn(MatmulTable) -> MatmulPlan,
) -> MatmulPlan {
    match selection {
        MatmulSelection::Float(choice) => float_plan(choice, table),
        MatmulSelection::Packed(choice) => packed_plan(choice, table),
        MatmulSelection::DequantizeThen { dequantize, choice } => {
            let plan = float_plan(choice, table);
            match dequantize {
                MatmulDequantize::Decode => Box::new(Then::new(Decoded, plan)),
                MatmulDequantize::Burn => Box::new(Then::new(FrameworkDequantized, plan)),
            }
        }
    }
}

/// The plan that runs `choice` on float inputs: a rule's kernel, then the fixed routines that
/// serve what it declines; a pin's kernel alone.
fn float_plan(choice: &MatmulChoice, table: &dyn Fn(MatmulTable) -> MatmulPlan) -> MatmulPlan {
    match choice {
        MatmulChoice::Autotune(chosen) => table(*chosen),
        MatmulChoice::Pin(kernel) => float_kernel(*kernel),
        MatmulChoice::Rule(kernel) => Box::new(InOrder::new(vec![
            float_kernel(*kernel),
            Box::new(RoutinePlan::at(
                CubeStagedMatmulRoutine,
                "cube-staged@default",
                CubeStagedMatmulRoutineSettings::DEFAULT,
            )),
            unit_registers(),
        ])),
    }
}

/// The plan that runs `choice` reading the packed inputs as they are stored.
fn packed_plan(
    choice: &MatmulPackedChoice,
    table: &dyn Fn(MatmulTable) -> MatmulPlan,
) -> MatmulPlan {
    match choice {
        MatmulPackedChoice::Autotune(chosen) => table(*chosen),
        MatmulPackedChoice::Pin(kernel) => packed_kernel(*kernel, table),
        MatmulPackedChoice::Rule { kernel, otherwise } => Box::new(InOrder::new(vec![
            packed_kernel(*kernel, table),
            planned(otherwise, table),
        ])),
    }
}

/// The routine a float launch of `kernel` runs.
fn float_kernel(kernel: MatmulKernel) -> MatmulPlan {
    match kernel {
        MatmulKernel::UnitRegisters => unit_registers(),
        MatmulKernel::PlaneShuffle => Box::new(RoutinePlan::first_served(
            PlaneShuffleMatmulRoutine,
            "plane-shuffle",
        )),
        MatmulKernel::Nvfp4TensorCores => unreachable!("a float product's selection refuses it"),
    }
}

/// What a launch of `kernel` runs reading the packed inputs as stored: the NVFP4 kernel is its
/// own table, whose candidates are its settings.
fn packed_kernel(kernel: MatmulKernel, table: &dyn Fn(MatmulTable) -> MatmulPlan) -> MatmulPlan {
    match kernel {
        MatmulKernel::PlaneShuffle => Box::new(RoutinePlan::first_served(
            PlaneShuffleMatmulRoutine,
            "plane-shuffle(packed as stored)",
        )),
        MatmulKernel::Nvfp4TensorCores => table(MatmulTable::Nvfp4),
        MatmulKernel::UnitRegisters => {
            unreachable!("a packed selection runs it on the decoded inputs")
        }
    }
}

/// The routine that asks nothing of the device, at its default: what serves any float product.
fn unit_registers() -> MatmulPlan {
    Box::new(RoutinePlan::at(
        UnitRegistersMatmulRoutine,
        "unit-registers@default",
        UnitRegistersMatmulRoutineSettings::DEFAULT,
    ))
}
