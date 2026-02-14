use clap::ValueEnum;
use core::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, Default)]
pub enum Target {
    HUFL,
    HULL,
    MUFL,
    MULL,
    LUFL,
    LULL,
    #[default]
    OT,
}
impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Target::HUFL => "HUFL",
            Target::HULL => "HULL",
            Target::MUFL => "MUFL",
            Target::MULL => "MULL",
            Target::LUFL => "LUFL",
            Target::LULL => "LULL",
            Target::OT => "OT",
        };
        write!(f, "{}", s)
    }
}
