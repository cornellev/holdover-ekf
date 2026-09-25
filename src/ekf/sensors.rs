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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn fix(lat: f64, lon: f64) -> GPSFix {
        GPSFix { t: 0.0, lat, lon, alt: 30.0 }
    }

    #[test]
    fn origin_is_zero() {
        let f = fix(37.4, -122.1);
        let enu = Origin::from_fix(&f).to_enu(&f);
        assert_relative_eq!(enu, Vec2::zeros(), epsilon=1e-9);
    }

    #[test]
    fn north_and_east_match_themselves() {
        let origin = Origin::from_fix(&fix(37.4, -122.1));

        // 1e-5 deg of lat ~ 1.11 m of north.
        let n = origin.to_enu(&fix(37.4 + 1e-5, -122.1));
        assert!(n[1] > 1.10 && n[1] < 1.12, "north = {}", n[1]);
        assert_relative_eq!(n[0], 0.0, epsilon=1e-6);

        //1e-5 deg of lon ~ 1.11 * cos(37.4 deg) ~ 0.88 m east
        let e = origin.to_enu(&fix(37.4, -122.1 + 1e-5));
        assert!(e[0] > 0.87 && e[0] < 0.90, "east = {}", e[0]);
        assert_relative_eq!(e[1], 0.0, epsilon=1e-6);
    }

    #[test]
    fn jacobian_is_same_as_h() {
        let x = Vec5::new(12.0, -7.0, 0.3, 9.0, 1e-3);
        assert_relative_eq!(gps_jacobian() * x, gps_h(&x), epsilon = 1e-12);
    }

}
