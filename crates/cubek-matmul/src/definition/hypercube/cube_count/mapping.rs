use cubecl::prelude::*;

use crate::definition::hypercube::{GlobalOrder, global_order::swizzle};

#[derive(CubeType, CubeLaunch)]
pub struct CubeMapping {
    strategy: CubeMappingStrategy,
    #[cube(comptime)]
    pub can_yield_extra_cubes: bool,
    #[cube(comptime)]
    global_order: GlobalOrder,
}

#[derive(CubeType, CubeLaunch)]
/// CubeCountPlan stripped of non-essential runtime information
///
/// This enum is given as runtime input to the matmul
#[allow(unused)] // Constructed via CubeMappingStrategyArgs only
pub(crate) enum CubeMappingStrategy {
    FromProblem,
    SmFirst {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    CubeFirst {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Flattened {
        m_cubes: u32,
        n_cubes: u32,
    },
    Spread {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
}

#[cube]
impl CubeMapping {
    /// Returns the number of valid cubes
    pub fn num_valid_cubes(&self) -> u32 {
        match &self.strategy {
            CubeMappingStrategy::FromProblem | CubeMappingStrategy::Flattened { .. } => {
                panic!("Shouldn't need to be called because the cube count should always be exact")
            }
            CubeMappingStrategy::SmFirst {
                m_cubes,
                n_cubes,
                batch_cubes,
            }
            | CubeMappingStrategy::CubeFirst {
                m_cubes,
                n_cubes,
                batch_cubes,
            }
            | CubeMappingStrategy::Spread {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
        }
    }

    /// Given a cube position, returns the tensor coordinates (m, n, batch).
    pub fn cube_pos_to_tensor_pos(&self) -> (u32, u32, u32) {
        match &self.strategy {
            CubeMappingStrategy::FromProblem => (CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z),

            CubeMappingStrategy::SmFirst {
                m_cubes, n_cubes, ..
            } => self.strategy.absolute_index_to_m_n_batch(
                CUBE_POS,
                *m_cubes,
                *n_cubes,
                self.global_order,
            ),

            CubeMappingStrategy::CubeFirst {
                m_cubes, n_cubes, ..
            } => self.strategy.absolute_index_to_m_n_batch(
                CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X,
                *m_cubes,
                *n_cubes,
                self.global_order,
            ),

            CubeMappingStrategy::Flattened { m_cubes, n_cubes } => self
                .strategy
                .absolute_index_to_m_n_batch(CUBE_POS_X, *m_cubes, *n_cubes, self.global_order),

            CubeMappingStrategy::Spread {
                m_cubes, n_cubes, ..
            } => self.strategy.absolute_index_to_m_n_batch(
                CUBE_POS,
                *m_cubes,
                *n_cubes,
                self.global_order,
            ),
        }
    }
}

#[cube]
impl CubeMappingStrategy {
    fn absolute_index_to_m_n_batch(
        &self,
        absolute_index: u32,
        m_cubes: u32,
        n_cubes: u32,
        #[comptime] global_order: GlobalOrder,
    ) -> (u32, u32, u32) {
        let batch_stride = m_cubes * n_cubes;
        let batch_pos = absolute_index / batch_stride;
        let matrix_pos = absolute_index % batch_stride;

        let (m_pos, n_pos) = match comptime!(global_order) {
            GlobalOrder::RowMajor => (matrix_pos / n_cubes, matrix_pos % n_cubes),
            GlobalOrder::ColMajor => (matrix_pos % m_cubes, matrix_pos / m_cubes),
            GlobalOrder::SwizzleRowMajor(w) => {
                let (x, y) = swizzle(matrix_pos, n_cubes, w);
                (y, x)
            }
            GlobalOrder::SwizzleColMajor(w) => swizzle(matrix_pos, m_cubes, w),
        };

        (m_pos, n_pos, batch_pos)
    }
}
