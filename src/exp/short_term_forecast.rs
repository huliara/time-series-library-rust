use crate::args::RootArgs;

pub struct ExpShortTermForecast {
    pub args: RootArgs,
}

impl crate::exp::Exp for ExpShortTermForecast {
    fn train(&mut self, _arg: &RootArgs) {
        // Implement training logic for short-term forecasting here
    }

    fn validate(&mut self, _arg: &RootArgs) {
        // Implement validation logic for short-term forecasting here
    }

    fn test(&mut self, _arg: &RootArgs) {
        // Implement testing logic for short-term forecasting here
    }
}
