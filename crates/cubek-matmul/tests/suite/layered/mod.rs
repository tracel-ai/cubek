use crate::suite::assert_result;
use crate::suite::layered::matmul_test_launcher::launch_matmul_algorithm;
use cubecl::Runtime;
use cubecl::TestRuntime;
use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::MatmulInputHandleRef;
use cubek_matmul::components::MatmulElems;
use cubek_matmul::components::MatmulIdent;
use cubek_matmul::components::MatmulProblem;
use cubek_matmul::components::MatmulSelection;
use cubek_matmul::components::MatrixLayout;
use cubek_matmul::components::SwizzleConfig;
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_matmul::components::{PartitionSize, StageSize, TileSize, TilingScheme};
use cubek_matmul::kernels::layered::simple::SimpleAlgorithm;
use cubek_matmul::kernels::layered::simple_unit::SimpleUnitAlgorithm;
use cubek_matmul::tune_key::MatmulElemType;
use cubek_std::test_utils::HostData;
use cubek_std::test_utils::HostDataType;
use cubek_std::test_utils::StrideSpec;
use cubek_std::test_utils::TestInput;
use cubek_std::test_utils::current_test_mode;

use crate::suite::layered::matmul_test_launcher::InputRepresentation;
use crate::suite::layered::matmul_test_launcher::test_matmul_algorithm;

pub mod matmul_test_launcher;

mod suite;
