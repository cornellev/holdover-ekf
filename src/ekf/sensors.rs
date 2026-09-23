use map_3d::{geodetic2enu, Ellipsoid};
use nalgebra::{SMatrix, SVector};

use crate::ekf::model::{S, Vec5}; // enum and vec structure
use crate::ekf::msg::GPSFix;

pub type Vec2 = SVector<f64, 2>;
pub type Mat2 = SMatrix<f64, 2, 2>;
pub type Mat2x5 = SMatrix<f64, 2, 5>;

pub struct Origin {
    lat: f64,
    lon: f64,
    alt: f64,
}

impl Origin {
    pub fn from_fix(fix: &GPSFix) -> Self {
        Self { lat: fix.lat.to_radians(), lon: fix.lon.to_radians(), alt: fix.alt }
    }
 
    pub fn to_enu(&self, fix: &GPSFix) -> Vec2 {
        let (e, n, _u) = geodetic2enu(
            fix.lat.to_radians(),
            fix.lon.to_radians(),
            fix.alt,
            self.lat,
            self.lon,
            self.alt,
            Ellipsoid::WGS84
        );
        Vec2::new(e,n)
    }
}

#[derive(Debug)]
pub struct GPSNoise {
    pub sigma_pos: f64,
}

impl Default for GPSNoise {
    fn default() -> Self { Self { sigma_pos: 2.5 } }
}

// Predicted measurement model for GPS
pub fn gps_h(x: &Vec5) -> Vec2 {
    Vec2::new(x[S::Pe], x[S::Pn])
}

pub fn gps_jacobian() -> Mat2x5 {
    let mut h = Mat2x5::zeros();
    h[(0, S::Pe.ix())] = 1.0;
    h[(1, S::Pn.ix())] = 1.0;
    h
}

pub fn gps_r(noise: &GPSNoise) -> Mat2 {
    Mat2::identity() * noise.sigma_pos.powi(2)
}
