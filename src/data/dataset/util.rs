use chrono::{Datelike, NaiveDateTime, Timelike};
use ndarray::Array2;

type FeatureFn = fn(&NaiveDateTime) -> f32;

fn second_of_minute(t: &NaiveDateTime) -> f32 {
    t.second() as f32 / 59.0 - 0.5
}

fn minute_of_hour(t: &NaiveDateTime) -> f32 {
    t.minute() as f32 / 59.0 - 0.5
}

fn hour_of_day(t: &NaiveDateTime) -> f32 {
    t.hour() as f32 / 23.0 - 0.5
}

fn day_of_week(t: &NaiveDateTime) -> f32 {
    t.weekday().num_days_from_monday() as f32 / 6.0 - 0.5
}

fn day_of_month(t: &NaiveDateTime) -> f32 {
    (t.day() as f32 - 1.0) / 30.0 - 0.5
}

fn day_of_year(t: &NaiveDateTime) -> f32 {
    (t.ordinal() as f32 - 1.0) / 365.0 - 0.5
}

fn month_of_year(t: &NaiveDateTime) -> f32 {
    (t.month() as f32 - 1.0) / 11.0 - 0.5
}

fn week_of_year(t: &NaiveDateTime) -> f32 {
    (t.iso_week().week() as f32 - 1.0) / 52.0 - 0.5
}

fn time_features_from_frequency_str(freq_str: &str) -> Vec<FeatureFn> {
    // Mapping based on pandas offset compatibility approximation
    let freq = if let Some(idx) = freq_str.find(|c: char| c.is_alphabetic()) {
        &freq_str[idx..]
    } else {
        freq_str
    };

    match freq {
        "h" | "H" => vec![hour_of_day, day_of_week, day_of_month, day_of_year],
        "t" | "T" | "min" => vec![
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
        "s" | "S" => vec![
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
        "d" | "D" => vec![day_of_week, day_of_month, day_of_year],
        "b" | "B" => vec![day_of_week, day_of_month, day_of_year],
        "w" | "W" => vec![day_of_month, week_of_year],
        "m" | "M" => vec![month_of_year],
        "a" | "A" | "y" | "Y" => vec![],
        _ => {
            // Check for contained strings to be more permissive like endswith logic
            if freq.ends_with('h') || freq.ends_with('H') {
                vec![hour_of_day, day_of_week, day_of_month, day_of_year]
            } else if freq.ends_with('t') || freq.ends_with('T') || freq.contains("min") {
                vec![
                    minute_of_hour,
                    hour_of_day,
                    day_of_week,
                    day_of_month,
                    day_of_year,
                ]
            } else if freq.ends_with('s') || freq.ends_with('S') {
                vec![
                    second_of_minute,
                    minute_of_hour,
                    hour_of_day,
                    day_of_week,
                    day_of_month,
                    day_of_year,
                ]
            } else if freq.ends_with('d')
                || freq.ends_with('D')
                || freq.ends_with('b')
                || freq.ends_with('B')
            {
                vec![day_of_week, day_of_month, day_of_year]
            } else if freq.ends_with('w') || freq.ends_with('W') {
                vec![day_of_month, week_of_year]
            } else if freq.ends_with('m') || freq.ends_with('M') {
                vec![month_of_year]
            } else {
                // default fallback similar to 'h' or just empty?
                // The python code raises error.
                panic!("Unsupported frequency {}", freq_str)
            }
        }
    }
}

pub fn time_features(dates: &[NaiveDateTime], freq: &str) -> Array2<f32> {
    let features = time_features_from_frequency_str(freq);
    let num_features = features.len();
    let num_samples = dates.len();

    let mut data = Vec::with_capacity(num_samples * num_features);

    for date in dates {
        for f in &features {
            data.push(f(date));
        }
    }

    // Result is (num_samples, num_features)
    Array2::from_shape_vec((num_samples, num_features), data).unwrap()
}
