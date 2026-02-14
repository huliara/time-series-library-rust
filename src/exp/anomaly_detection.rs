use crate::args::RootArgs;
use crate::exp::Exp;
pub struct ExpAnomalyDetection {
    pub args: RootArgs,
}
impl Exp for ExpAnomalyDetection {
    fn train(&mut self, _arg: &RootArgs) {
        // Training logic for anomaly detection
    }

    fn validate(&mut self, _arg: &RootArgs) {
        // Validation logic for anomaly detection
    }

    fn test(&mut self, _arg: &RootArgs) {
        // Testing logic for anomaly detection
    }
}
