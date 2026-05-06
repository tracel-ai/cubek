use crate::registry::CatalogEntry;

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

pub fn strategies() -> Vec<CatalogEntry<CopyStrategyEnum>> {
    vec![
        CatalogEntry::new(
            "dummy",
            "Dummy (all units copy duplicatively)",
            CopyStrategyEnum::DummyCopy,
        ),
        CatalogEntry::new(
            "coalesced",
            "Coalesced (sync)",
            CopyStrategyEnum::CoalescedCopy,
        ),
        CatalogEntry::new(
            "single_duplicated_all",
            "Single slice / duplicated all units",
            CopyStrategyEnum::MemcpyAsyncSingleSliceDuplicatedAll,
        ),
        CatalogEntry::new(
            "single_elected",
            "Single slice / elected unit",
            CopyStrategyEnum::MemcpyAsyncSingleSliceElected,
        ),
        CatalogEntry::new(
            "single_elected_cooperative",
            "Single slice / elected cooperative",
            CopyStrategyEnum::MemcpyAsyncSingleSliceElectedCooperative,
        ),
        CatalogEntry::new(
            "split_plane_duplicated_unit",
            "Split per plane / duplicated units",
            CopyStrategyEnum::MemcpyAsyncSplitPlaneDuplicatedUnit,
        ),
        CatalogEntry::new(
            "split_plane_elected_unit",
            "Split per plane / elected unit",
            CopyStrategyEnum::MemcpyAsyncSplitPlaneElectedUnit,
        ),
        CatalogEntry::new(
            "split_duplicated_all",
            "Split / duplicated all",
            CopyStrategyEnum::MemcpyAsyncSplitDuplicatedAll,
        ),
        CatalogEntry::new(
            "split_large_unit_with_idle",
            "Split / large unit with idle",
            CopyStrategyEnum::MemcpyAsyncSplitLargeUnitWithIdle,
        ),
        CatalogEntry::new(
            "split_small_unit_coalesced_loop",
            "Split / small unit coalesced loop",
            CopyStrategyEnum::MemcpyAsyncSplitSmallUnitCoalescedLoop,
        ),
        CatalogEntry::new(
            "split_medium_unit_coalesced_once",
            "Split / medium unit coalesced once",
            CopyStrategyEnum::MemcpyAsyncSplitMediumUnitCoalescedOnce,
        ),
    ]
}
