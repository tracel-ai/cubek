//! Kernel Test Workflow
//!
//! 1. **Execution**  
//!    - Kernel runs or fails to compile [`ExecutionOutcome`].
//!      - `Executed`: ran (correctness not checked).  
//!      - `CompileError`: did not compile.
//!
//! 2. **Validation**  
//!    - Check correctness of the executed kernel [`ValidationResult`].
//!      - `Pass`: result matches reference.  
//!      - `Fail`: result incorrect.  
//!      - `Skipped`: could not decide.
//!
//! 3. **Test Outcome**  
//!    - Combines execution + validation [`TestOutcome`].
//!
//! 4. **Policy Decision**  
//!    - Applies test mode to decide if the test passes [`TestDecision`].
//!      - `Accept`: test passes.  
//!      - `Reject(String)`: test fails.  
//!    - Call [`TestDecision::enforce`] to actually fail the test.

mod base;
mod result;

pub use base::*;
pub use result::*;
