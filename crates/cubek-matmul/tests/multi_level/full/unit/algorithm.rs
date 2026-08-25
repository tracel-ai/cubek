mod simple {
    use super::*;
    use super::launch_simple as launch;

    include!("precision.rs");
}

mod double_buffering {
    use super::*;
    use super::launch_double_buffering as launch;

    include!("precision.rs");
}

mod interleaved {
    use super::*;
    use super::launch_interleaved as launch;

    include!("precision.rs");
}
