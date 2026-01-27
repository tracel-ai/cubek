use cubecl::prelude::*;

use crate::components::global::specialization::config::LoadingSides;
use crate::components::global::specialization::roles::PlaneFlowPartitionRule;
use crate::components::global::{PlaneFlowConfig, SpecializedLoadingSides};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Comptime information of specializer
pub enum SpecializerKind {
    Specialized {
        main_flow_loading_side: LoadingSides,
        load_only_loading_side: LoadingSides,
        role_rule_config: PlaneFlowPartitionRule,
    },
    NotSpecialized,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Specialization information in cube functions
pub struct Specializer {
    #[cube(comptime)]
    pub kind: SpecializerKind,
}

#[cube]
impl Specializer {
    pub fn new(
        #[comptime] plane_flow_config: PlaneFlowConfig,
        #[comptime] loading_sides: SpecializedLoadingSides,
    ) -> Specializer {
        if plane_flow_config.has_specialization() {
            Specializer {
                kind: comptime! {
                    SpecializerKind::Specialized {
                        main_flow_loading_side: loading_sides.main_flow,
                        load_only_loading_side: loading_sides.load_only,
                        role_rule_config: plane_flow_config.partition_rule
                    }
                },
            }
        } else {
            Specializer {
                kind: comptime! {SpecializerKind::NotSpecialized},
            }
        }
    }
}
