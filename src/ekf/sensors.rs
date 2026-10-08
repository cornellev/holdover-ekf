use map_3d::{geodetic2enu, Ellipsoid};
use nalgebra::{SMatrix, SVector};

use crate::ekf::model::{StateEnum, Vec5}; // enum and vec structure
use sensor_defs::GPS;

pub type Vec1 = SVector<f64, 1>;
pub type Mat1 = SMatrix<f64, 1, 1>;
pub type Mat1x5 = SMatrix<f64, 1, 5>;
pub type Vec2 = SVector<f64, 2>;
pub type Mat2 = SMatrix<f64, 2, 2>;
pub type Mat2x5 = SMatrix<f64, 2, 5>;

#[derive(Debug, Clone, Copy)] //NOTE: we don't need clone/copy as this is readable ref only, right?
pub struct Origin {
    lat: f64,
    lon: f64,
    alt: f64,
}

impl Origin {
    pub fn from_fix(fix: &GPS) -> Self {
        Self { lat: fix.lat_deg().to_radians(), lon: fix.lon_deg().to_radians(), alt: fix.alt_m() }
    }
 
    pub fn to_enu(&self, fix: &GPS) -> Vec2 {
        let (e, n, _u) = geodetic2enu(
            fix.lat_deg().to_radians(),
            fix.lon_deg().to_radians(),
            fix.alt_m(),
            self.lat,
            self.lon,
            self.alt,
            Ellipsoid::WGS84
        );
        Vec2::new(e,n)
    }
}

/// Predicted measurement model for GPS
pub fn gps_h(x: &Vec5) -> Vec2 {
    Vec2::new(x[StateEnum::Pe], x[StateEnum::Pn])
}

pub fn gps_jacobian() -> Mat2x5 {
    let mut h = Mat2x5::zeros();
    h[(0, StateEnum::Pe.ix())] = 1.0;
    h[(1, StateEnum::Pn.ix())] = 1.0;
    h
}

///R from the receiver's own per-fix accuracy: diag(var_e, var_n).
pub fn gps_r(fix: &GPS) -> Mat2 {
    let var = fix.pos_var_m2();
    Mat2::from_diagonal(&Vec2::new(var[0], var[1]))
}

///Variance of a fix's horizontal error projected onto unit vector `u` (ENU convention).
pub fn gps_var_along(var_m2: [f64; 3], u: &Vec2) -> f64 {
    u[0].powi(2) * var_m2[0] + u[1].powi(2) * var_m2[1]
}

#[derive(Debug, Clone, Copy)]
pub struct IMUNoise {
    pub sigma_ay: f64 // accelerometer not gyro
}

impl Default for IMUNoise {
    fn default() -> Self { Self { sigma_ay: 0.5 } }
}

pub fn imu_h(x: &Vec5, omega_z: f64) -> Vec1 {
    Vec1::new(x[StateEnum::V] * (omega_z - x[StateEnum::Bw]))
}

pub fn imu_jacobian(x: &Vec5, omega_z: f64) -> Mat1x5 {
    let mut h = Mat1x5::zeros();
    h[(0, StateEnum::V.ix())] = omega_z - x[StateEnum::Bw];
    h[(0, StateEnum::Bw.ix())] = -x[StateEnum::V];
    h
}

pub fn imu_r(noise: &IMUNoise) -> Mat1 {
    Mat1::new(noise.sigma_ay.powi(2))
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn fix(lat: f64, lon: f64) -> GPS {
        // GPSFix { t: 0.0, lat, lon, alt: 30.0 }
        GPS::new(0, lat, lon, 30.0, [2.0; 3]).unwrap()
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
