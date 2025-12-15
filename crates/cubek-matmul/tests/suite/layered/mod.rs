use crate::suite::assert_result;
use crate::suite::layered::matmul_test_launcher::launch_matmul_algorithm;
use cubecl::Runtime;
use cubecl::TestRuntime;
use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_matmul::launch::MatmulElems;
use cubek_matmul::launch::MatmulIdent;
use cubek_matmul::launch::MatmulProblem;
use cubek_matmul::launch::MatmulSelection;
use cubek_matmul::launch::MatrixLayout;
use cubek_matmul::launch::SwizzleConfig;
use cubek_matmul::launch::{PartitionSize, StageSize, TileSize, TilingScheme};
use cubek_matmul::routines::simple::SimpleAlgorithm;
use cubek_matmul::routines::simple_unit::SimpleUnitAlgorithm;
use cubek_matmul::tune_key::MatmulElemType;
use cubek_test_utils::HostData;
use cubek_test_utils::HostDataType;
use cubek_test_utils::StrideSpec;
use cubek_test_utils::TestInput;
use cubek_test_utils::current_test_mode;

use crate::suite::layered::matmul_test_launcher::InputRepresentation;
use crate::suite::layered::matmul_test_launcher::test_matmul_algorithm;

pub mod matmul_test_launcher;

mod suite;
