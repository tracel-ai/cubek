mod layout;
mod fft;

pub use fft::*;

//pub struct SignalSpec {
//    pub signal_duration: f32,
//    pub channels: usize,
//    pub sample_rate: usize,
//    pub window_length: usize,
//    pub hop_length: usize,
//}
//
//impl SignalSpec {
//    pub fn signal_shape(&self) -> [usize; 3] {
//        let total_samples = (self.signal_duration * self.sample_rate as f32).ceil() as usize;
//        let num_windows = total_samples.div_ceil(self.hop_length);
//        [num_windows, self.channels, self.window_length]
//    }
//
//    pub fn spectrum_shape(&self) -> [usize; 3] {
//        let total_samples = (self.signal_duration * self.sample_rate as f32).ceil() as usize;
//        let num_windows = total_samples.div_ceil(self.hop_length);
//        let num_frequency_bins = self.window_length / 2 + 1;
//        [num_windows, self.channels, num_frequency_bins]
//    }
//}
