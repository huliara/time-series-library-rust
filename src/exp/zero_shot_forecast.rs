use crate::args::RootArgs;
pub struct ExpZeroShotForecast {
    pub args: RootArgs,
}

impl crate::exp::Exp for ExpZeroShotForecast {
    fn train(&mut self, _arg: &RootArgs) {
        // Implement training logic for zero-shot forecasting here
    }

    fn validate(&mut self, _arg: &RootArgs) {
        // Implement validation logic for zero-shot forecasting here
    }

    fn test(&mut self, _arg: &RootArgs) {
        // Implement testing logic for zero-shot forecasting here
    }
}
