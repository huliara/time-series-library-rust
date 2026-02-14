use clap::ValueEnum;
use core::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize)]
pub enum TimeEmbed {
    TimeF,
    Fixed,
}
impl fmt::Display for TimeEmbed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TimeEmbed::TimeF => "timeF",
            TimeEmbed::Fixed => "fixed",
        };
        write!(f, "{}", s)
    }
}
