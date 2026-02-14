use crate::args::RootArgs;
use crate::exp::Exp;
pub struct ExpClassification {
    pub args: RootArgs,
}
impl Exp for ExpClassification {
    fn train(&mut self, _arg: &RootArgs) {
        // Implement training logic here
    }

    fn validate(&mut self, _arg: &RootArgs) {
        // Implement validation logic here
    }

    fn test(&mut self, _arg: &RootArgs) {
        // Implement testing logic here
    }
}
