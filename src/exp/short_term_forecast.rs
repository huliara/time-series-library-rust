use crate::args::Args;

pub struct ExpShortTermForecast {
    pub args: Args,
}

impl crate::exp::Exp for ExpShortTermForecast {
    fn train(&mut self, _arg: &Args) {
        // Implement training logic for short-term forecasting here
    }

    fn validate(&mut self, _arg: &Args) {
        // Implement validation logic for short-term forecasting here
    }

    fn test(&mut self, _arg: &Args) {
        // Implement testing logic for short-term forecasting here
    }
}
