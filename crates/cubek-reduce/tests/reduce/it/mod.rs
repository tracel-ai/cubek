//! The integration tests: the `reduce_dim` matrix ([`matrix`]) and the cases it leaves out.

mod test_case;

mod argtopk_shared_memory;
mod empty_output;
mod logical;
mod matrix;
mod nan_extrema;
mod plane_reduction;
mod strategies;
mod topk_negative_infinity;
mod topk_unroll_limit;
mod topk_vectorized_output;
mod topk_with_indices_cube;
mod topk_with_indices_hypercube;
mod with_indices_validation;
