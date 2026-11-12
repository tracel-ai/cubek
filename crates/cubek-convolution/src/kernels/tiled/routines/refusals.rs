use cubecl::tile::launch::Refusal;

/// What a setting is declined with where the device offers no instruction of its kind for the
/// problem's elements.
pub const CONV_INSTRUCTION_UNOFFERED: Refusal = Refusal::Declined {
    why: "the device offers no instruction of this kind for the problem's elements",
};

/// What a setting is declined with where it asks for a warpgroup MMA, whose stages only the TMA
/// engine lands and which this convolution's fill never does.
pub const CONV_WGMMA_UNSERVED: Refusal = Refusal::Declined {
    why: "a warpgroup MMA reads its stages as the TMA engine lands them",
};

/// What a launch is declined with where it pins a cube staged setting on the unit registers
/// routine, whose settings are its own.
pub const CONV_SETTINGS_NOT_UNIT: Refusal = Refusal::Declined {
    why: "the unit registers routine takes none of the cube staged settings",
};

/// What a launch is declined with where it pins a setting on the autotune, which chooses its own.
pub const CONV_SETTINGS_NOT_TUNED: Refusal = Refusal::Declined {
    why: "the tuned routine chooses its setting, so it takes none",
};
