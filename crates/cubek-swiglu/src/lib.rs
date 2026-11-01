//! SwiGLU, `silu(gate) ⊙ up` over a fused `[gate | up]` row, in `f32` whatever the rows' element:
//! its routines, their selection and autotune ([`kernels`]). Its microkernels are
//! cubek-microkernel's (`cubek_microkernel::swiglu`).

pub mod kernels;
