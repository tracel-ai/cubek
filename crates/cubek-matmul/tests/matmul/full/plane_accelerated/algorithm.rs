mod simple_cyclic {
    use super::*;
    use super::launch_simple_cyclic as launch;

    include!("precision.rs");
}

mod simple_strided {
    use super::*;
    use super::launch_simple_strided as launch;

    include!("precision.rs");
}

mod simple_tilewise {
    use super::*;
    use super::launch_simple_tilewise as launch;

    include!("precision.rs");
}

mod simple_barrier_cooperative {
    use super::*;
    use super::launch_simple_barrier_cooperative as launch;

    include!("precision.rs");
}

mod simple_barrier_cyclic {
    use super::*;
    use super::launch_simple_barrier_cyclic as launch;

    include!("precision.rs");
}

mod double_buffering_cyclic {
    use super::*;
    use super::launch_double_buffering_cyclic as launch;

    include!("precision.rs");
}

mod double_buffering_tilewise {
    use super::*;
    use super::launch_double_buffering_tilewise as launch;

    include!("precision.rs");
}

mod double_buffering_hybrid {
    use super::*;
    use super::launch_double_buffering_hybrid as launch;

    include!("precision.rs");
}

mod ordered_double_buffering {
    use super::*;
    use super::launch_ordered_double_buffering as launch;

    include!("precision.rs");
}

mod specialized_cyclic {
    use super::*;
    use super::launch_specialized_cyclic as launch;

    include!("precision.rs");
}

mod specialized_strided {
    use super::*;
    use super::launch_specialized_strided as launch;

    include!("precision.rs");
}
