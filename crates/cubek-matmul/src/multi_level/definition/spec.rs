use cubecl::prelude::*;

use crate::{definition::MatmulIdent, multi_level::components::global::memory::ViewDirection};

/// Matrix multiplication precisions.
pub trait MatmulTypes: Send + Sync + Copy + 'static {
    /// Element type of lhs input tensor of the kernel.
    type Lhs: MatrixTypes;
    /// Element type of rhs input tensor of the kernel.
    type Rhs: MatrixTypes;
    /// Element type of acc input tensor of the kernel.
    type Acc: MatrixTypes;
}

pub trait MatrixTypes: Send + Sync + Copy + 'static {
    /// Element type of input tensor in global memory
    type Global: Numeric;
    type GlobalSize: Size;
    /// Element type once stored in shared memory
    type Stage: Numeric;
    type StageSize: Size;
    /// Element type once in registers for computation
    type Register: Numeric;
    type RegisterSize: Size;
}

impl<EG: Numeric, SG: Size, ES: Numeric, SS: Size, ER: Numeric, SR: Size> MatrixTypes
    for (EG, SG, ES, SS, ER, SR)
{
    type Global = EG;
    type GlobalSize = SG;
    type Stage = ES;
    type StageSize = SS;
    type Register = ER;
    type RegisterSize = SR;
}

impl<Lhs: MatrixTypes, Rhs: MatrixTypes, Acc: MatrixTypes> MatmulTypes for (Lhs, Rhs, Acc) {
    type Lhs = Lhs;
    type Rhs = Rhs;
    type Acc = Acc;
}

pub type Lhs<MT> = <MT as MatmulTypes>::Lhs;
pub type Rhs<MT> = <MT as MatmulTypes>::Rhs;
pub type Acc<MT> = <MT as MatmulTypes>::Acc;

pub type Global<MT> = <MT as MatrixTypes>::Global;
pub type GlobalSize<MT> = <MT as MatrixTypes>::GlobalSize;

pub type Stage<MT> = <MT as MatrixTypes>::Stage;
pub type StageSize<MT> = <MT as MatrixTypes>::StageSize;

pub type Register<MT> = <MT as MatrixTypes>::Register;
pub type RegisterSize<MT> = <MT as MatrixTypes>::RegisterSize;

// ==================== LHS ====================

// Vector forms
pub type LhsG<MT> = Vector<Global<Lhs<MT>>, GlobalSize<Lhs<MT>>>;
pub type LhsS<MT> = Vector<Stage<Lhs<MT>>, StageSize<Lhs<MT>>>;
pub type LhsR<MT> = Vector<Register<Lhs<MT>>, RegisterSize<Lhs<MT>>>;

// Element / Size splits
pub type LhsGE<MT> = <Lhs<MT> as MatrixTypes>::Global;
pub type LhsGS<MT> = <Lhs<MT> as MatrixTypes>::GlobalSize;

pub type LhsSE<MT> = <Lhs<MT> as MatrixTypes>::Stage;
pub type LhsSS<MT> = <Lhs<MT> as MatrixTypes>::StageSize;

pub type LhsRE<MT> = <Lhs<MT> as MatrixTypes>::Register;
pub type LhsRS<MT> = <Lhs<MT> as MatrixTypes>::RegisterSize;

// ==================== RHS ====================

// Vector forms
pub type RhsG<MT> = Vector<Global<Rhs<MT>>, GlobalSize<Rhs<MT>>>;
pub type RhsS<MT> = Vector<Stage<Rhs<MT>>, StageSize<Rhs<MT>>>;
pub type RhsR<MT> = Vector<Register<Rhs<MT>>, RegisterSize<Rhs<MT>>>;

// Element / Size splits
pub type RhsGE<MT> = <Rhs<MT> as MatrixTypes>::Global;
pub type RhsGS<MT> = <Rhs<MT> as MatrixTypes>::GlobalSize;

pub type RhsSE<MT> = <Rhs<MT> as MatrixTypes>::Stage;
pub type RhsSS<MT> = <Rhs<MT> as MatrixTypes>::StageSize;

pub type RhsRE<MT> = <Rhs<MT> as MatrixTypes>::Register;
pub type RhsRS<MT> = <Rhs<MT> as MatrixTypes>::RegisterSize;

// ==================== ACC ====================

// Vector forms
pub type AccG<MT> = Vector<Global<Acc<MT>>, GlobalSize<Acc<MT>>>;
pub type AccS<MT> = Vector<Stage<Acc<MT>>, StageSize<Acc<MT>>>;
pub type AccR<MT> = Vector<Register<Acc<MT>>, RegisterSize<Acc<MT>>>;

// Element / Size splits
pub type AccGE<MT> = <Acc<MT> as MatrixTypes>::Global;
pub type AccGS<MT> = <Acc<MT> as MatrixTypes>::GlobalSize;

pub type AccSE<MT> = <Acc<MT> as MatrixTypes>::Stage;
pub type AccSS<MT> = <Acc<MT> as MatrixTypes>::StageSize;

pub type AccRE<MT> = <Acc<MT> as MatrixTypes>::Register;
pub type AccRS<MT> = <Acc<MT> as MatrixTypes>::RegisterSize;

impl MatmulIdent {
    pub fn view_direction(&self) -> ViewDirection {
        match self {
            MatmulIdent::Lhs => ViewDirection::Col,
            MatmulIdent::Rhs => ViewDirection::Row,
            MatmulIdent::Out => ViewDirection::None,
        }
    }
}
