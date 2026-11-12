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
