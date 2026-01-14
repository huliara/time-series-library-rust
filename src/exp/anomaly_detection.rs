use crate::args::Args;
use crate::exp::Exp;
pub struct ExpAnomalyDetection {
    pub args: Args,
}
impl Exp for ExpAnomalyDetection {
    fn train(&mut self, _arg: &Args) {
        // Training logic for anomaly detection
    }

    fn validate(&mut self, _arg: &Args) {
        // Validation logic for anomaly detection
    }

    fn test(&mut self, _arg: &Args) {
        // Testing logic for anomaly detection
    }
}
