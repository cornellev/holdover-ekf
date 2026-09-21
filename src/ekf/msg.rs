use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IMU {
    pub t: f64,
    pub gyro_z: f64,
    pub accel_x: f64,
    pub accel_y: f64
}

//NOTE: shouldn't we only be using mutable references here, not copying?
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GPSFix {
    pub t: f64,
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}
