use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ItemDescriptor {
    pub id: &'static str,
    pub label: &'static str,
}

#[derive(Debug, Clone)]
pub struct RunSamples {
    pub durations: Vec<Duration>,
}
