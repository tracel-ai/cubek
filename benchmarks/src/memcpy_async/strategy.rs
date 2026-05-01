use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_DUMMY: &str = "dummy";
pub const STRATEGY_COALESCED: &str = "coalesced";
pub const STRATEGY_SINGLE_DUPLICATED_ALL: &str = "single_duplicated_all";
pub const STRATEGY_SINGLE_ELECTED: &str = "single_elected";
pub const STRATEGY_SINGLE_ELECTED_COOPERATIVE: &str = "single_elected_cooperative";
pub const STRATEGY_SPLIT_PLANE_DUPLICATED_UNIT: &str = "split_plane_duplicated_unit";
pub const STRATEGY_SPLIT_PLANE_ELECTED_UNIT: &str = "split_plane_elected_unit";
pub const STRATEGY_SPLIT_DUPLICATED_ALL: &str = "split_duplicated_all";
pub const STRATEGY_SPLIT_LARGE_UNIT_WITH_IDLE: &str = "split_large_unit_with_idle";
pub const STRATEGY_SPLIT_SMALL_UNIT_COALESCED_LOOP: &str = "split_small_unit_coalesced_loop";
pub const STRATEGY_SPLIT_MEDIUM_UNIT_COALESCED_ONCE: &str = "split_medium_unit_coalesced_once";

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum CopyStrategyEnum {
    DummyCopy,
    CoalescedCopy,
    MemcpyAsyncSingleSliceDuplicatedAll,
    MemcpyAsyncSingleSliceElected,
    MemcpyAsyncSingleSliceElectedCooperative,
    MemcpyAsyncSplitPlaneDuplicatedUnit,
    MemcpyAsyncSplitPlaneElectedUnit,
    MemcpyAsyncSplitDuplicatedAll,
    MemcpyAsyncSplitLargeUnitWithIdle,
    MemcpyAsyncSplitSmallUnitCoalescedLoop,
    MemcpyAsyncSplitMediumUnitCoalescedOnce,
}

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_DUMMY.to_string(),
            label: "Dummy (all units copy duplicatively)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_COALESCED.to_string(),
            label: "Coalesced (sync)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SINGLE_DUPLICATED_ALL.to_string(),
            label: "Single slice / duplicated all units".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SINGLE_ELECTED.to_string(),
            label: "Single slice / elected unit".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SINGLE_ELECTED_COOPERATIVE.to_string(),
            label: "Single slice / elected cooperative".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SPLIT_PLANE_DUPLICATED_UNIT.to_string(),
            label: "Split per plane / duplicated units".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SPLIT_PLANE_ELECTED_UNIT.to_string(),
            label: "Split per plane / elected unit".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SPLIT_DUPLICATED_ALL.to_string(),
            label: "Split / duplicated all".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SPLIT_LARGE_UNIT_WITH_IDLE.to_string(),
            label: "Split / large unit with idle".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SPLIT_SMALL_UNIT_COALESCED_LOOP.to_string(),
            label: "Split / small unit coalesced loop".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SPLIT_MEDIUM_UNIT_COALESCED_ONCE.to_string(),
            label: "Split / medium unit coalesced once".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<CopyStrategyEnum> {
    Some(match id {
        STRATEGY_DUMMY => CopyStrategyEnum::DummyCopy,
        STRATEGY_COALESCED => CopyStrategyEnum::CoalescedCopy,
        STRATEGY_SINGLE_DUPLICATED_ALL => CopyStrategyEnum::MemcpyAsyncSingleSliceDuplicatedAll,
        STRATEGY_SINGLE_ELECTED => CopyStrategyEnum::MemcpyAsyncSingleSliceElected,
        STRATEGY_SINGLE_ELECTED_COOPERATIVE => {
            CopyStrategyEnum::MemcpyAsyncSingleSliceElectedCooperative
        }
        STRATEGY_SPLIT_PLANE_DUPLICATED_UNIT => {
            CopyStrategyEnum::MemcpyAsyncSplitPlaneDuplicatedUnit
        }
        STRATEGY_SPLIT_PLANE_ELECTED_UNIT => CopyStrategyEnum::MemcpyAsyncSplitPlaneElectedUnit,
        STRATEGY_SPLIT_DUPLICATED_ALL => CopyStrategyEnum::MemcpyAsyncSplitDuplicatedAll,
        STRATEGY_SPLIT_LARGE_UNIT_WITH_IDLE => CopyStrategyEnum::MemcpyAsyncSplitLargeUnitWithIdle,
        STRATEGY_SPLIT_SMALL_UNIT_COALESCED_LOOP => {
            CopyStrategyEnum::MemcpyAsyncSplitSmallUnitCoalescedLoop
        }
        STRATEGY_SPLIT_MEDIUM_UNIT_COALESCED_ONCE => {
            CopyStrategyEnum::MemcpyAsyncSplitMediumUnitCoalescedOnce
        }
        _ => return None,
    })
}
