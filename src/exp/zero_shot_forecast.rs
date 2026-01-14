use crate::args::Args;
pub struct ExpZeroShotForecast {
    pub args: Args,
}

impl crate::exp::Exp for ExpZeroShotForecast {
    fn train(&mut self, _arg: &Args) {
        // Implement training logic for zero-shot forecasting here
    }

    fn validate(&mut self, _arg: &Args) {
        // Implement validation logic for zero-shot forecasting here
    }

    fn test(&mut self, _arg: &Args) {
        // Implement testing logic for zero-shot forecasting here
    }
}
