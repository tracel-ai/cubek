mod simple_cyclic {
    use super::*;
    use super::launch_simple_cyclic as launch;

    include!("precision.rs");
}

mod double_buffering_cyclic {
    use super::*;
    use super::launch_double_buffering_cyclic as launch;

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
