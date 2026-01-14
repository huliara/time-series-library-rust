use crate::args::Args;
use crate::exp::Exp;
pub struct ExpClassification {
    pub args: Args,
}
impl Exp for ExpClassification {
    fn train(&mut self, _arg: &Args) {
        // Implement training logic here
    }

    fn validate(&mut self, _arg: &Args) {
        // Implement validation logic here
    }

    fn test(&mut self, _arg: &Args) {
        // Implement testing logic here
    }
}
