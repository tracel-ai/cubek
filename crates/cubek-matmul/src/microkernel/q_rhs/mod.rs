//! The product against a packed weight: its cube shuffle ([`cube`]), each box it cuts made of the
//! product's [step](super::MatmulStep), whose planes are the float product's plane fragment, which
//! decodes the weight as its fragments load, or its plane shuffle, which scales the weight as it
//! reads it. The cube that stages is the float product's
//! ([`CubeStagedMicrokernel`](super::dense::cube::CubeStagedMicrokernel)), its stages decoding the
//! weight.

pub mod cube;
