mod simple_tma {
    use super::*;
    use super::launch_simple_tma as launch;

    include!("precision.rs");
}

mod double_buffering_tma {
    use super::*;
    use super::launch_double_buffering_tma as launch;

    include!("precision.rs");
}

mod specialized_tma {
    use super::*;
    use super::launch_specialized_tma as launch;

    include!("precision.rs");
}
