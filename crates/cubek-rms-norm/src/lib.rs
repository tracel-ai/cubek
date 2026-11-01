//! The RMS norm, `x / sqrt(mean(x², last) + eps) · weight`, the square-sum in `f32` whatever the
//! rows' element: its routines, their selection and autotune ([`kernels`]). Its microkernels are
//! cubek-microkernel's (`cubek_microkernel::rms_norm`).

pub mod kernels;
