use nalgebra::{SMatrix, SVector};

use crate::ekf::filter::update;
use crate::ekf::model::{predict, Mat5, ProcessNoise, Vec5};
use crate::ekf::msg::{GPSFix, IMU};
use crate::ekf::sensors::{
    gps_h, gps_jacobian, gps_r, imu_h, imu_jacobian, imu_r, GPSNoise, IMUNoise, Origin, Vec1, Vec2
};

#[derive(Debug, Clone, Copy)]
pub struct Tuning {
    pub q: ProcessNoise,
    pub gps: GPSNoise,
    pub imu: IMUNoise,
    pub sigma_b0: f64, // initial gyro bias std [rad/s]
    pub min_baseline: f64, // metres driven before heading is initalized
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            q: ProcessNoise::default(),
            gps: GPSNoise::default(),
            imu: IMUNoise::default(),
            sigma_b0: 1e-2,
            min_baseline: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reject {
    NotRunning,
    OutOfOrder { dt: f64 },
    SingularInnovation,
}

#[derive(Debug)]
struct Running {
    origin: Origin,
    x: Vec5,
    p: Mat5,
    t: f64,
}

impl Running {
    // Heading frmo the chord between two fixes; variance from sigma_gps at each end.
    fn from_two_fixes(origin: Origin, first: Vec2, t0: f64, z: Vec2, t: f64, tun: &Tuning) -> Self {
        let d = z - first;
        let (len, dt) = (d.norm(), t - t0);
        let var_gps = tun.gps.sigma_pos.powi(2);

        let x = Vec5::new(z[0], z[1], d[1].atan2(d[0]), len / dt, 0.0);
        let p = Mat5::from_diagonal(&Vec5::new(
            var_gps,
            var_gps,
            2.0 * var_gps / len.powi(2),
            2.0 * var_gps / dt.powi(2),
            tun.sigma_b0.powi(2)
        ));

        Self { origin, x, p, t}
    }

    fn advance_to(&mut self, t: f64, gyro_z: f64, q: &ProcessNoise) -> Result<(), Reject> {
        let dt = t - self.t;
        if dt < 0.0 {
            return Err(Reject::OutOfOrder { dt });
        }
        if dt > 0.0 {
            (self.x, self.p) = predict(&self.x, &self.p, gyro_z, dt, q);
            self.t = t;
        }
        Ok(())
    }

    fn correct<const M: usize>(
        &mut self,
        z: &SVector<f64, M>,
        z_hat: &SVector<f64, M>,
        h: &SMatrix<f64, M, 5>,
        r: &SMatrix<f64, M, M>,
    ) -> Result<(), Reject> {
        (self.x, self.p) =
            update(&self.x, &self.p, z, z_hat, h, r).ok_or(Reject::SingularInnovation)?;
        Ok(())
    }
}

#[derive(Debug)]
#[expect(clippy::large_enum_variant, reason = "one long-lived instance")]
enum Phase {
    AwaitingFix,
    AwaitingMotion { origin: Origin, first: Vec2, t0: f64 },
    Running(Running),
}

#[derive(Debug)]
pub struct EKF {
    tuning: Tuning,
    gyro_z: f64, // ZOH on last gyro sample
    phase: Phase,
}

impl EKF {
    pub fn new(tuning: Tuning) -> Self {
        Self { tuning, gyro_z: 0.0, phase: Phase::AwaitingFix }
    }

    pub fn estimate(&self) -> Option<(&Vec5, &Mat5)> {
        match &self.phase {
            Phase::Running(run) => Some((&run.x, &run.p)),
            _ => None,
        }
    }

    pub fn on_imu(&mut self, imu: &IMU) -> Result<(), Reject> {
        let Phase::Running(run) = &mut self.phase else {
            self.gyro_z = imu.gyro_z;
            return Err(Reject::NotRunning);
        };

        // Hold the *previous* gyro smaple over [t_prev, imu.t] then latch onto the new one when ready
        run.advance_to(imu.t, self.gyro_z, &self.tuning.q)?;
        self.gyro_z = imu.gyro_z;

        run.correct(
            &Vec1::new(imu.accel_y),
            &imu_h(&run.x, imu.gyro_z),
            &imu_jacobian(&run.x, imu.gyro_z),
            &imu_r(&self.tuning.imu),
        )
    }

    pub fn on_gps(&mut self, fix: &GPSFix) -> Result<(), Reject> {
        match &mut self.phase {
            Phase::AwaitingFix => {
                self.phase = Phase::AwaitingMotion {
                    origin: Origin::from_fix(fix),
                    first: Vec2::zeros(),
                    t0: fix.t,
                };
                Ok(())
            }
            Phase::AwaitingMotion { origin, first, t0} => {
                let z = origin.to_enu(fix);
                if (z - *first).norm() >= self.tuning.min_baseline && fix.t > *t0 {
                    let run = Running::from_two_fixes(*origin, *first, *t0, z, fix.t, &self.tuning);
                    self.phase = Phase::Running(run);
                }
                Ok(())
            }
            Phase::Running(run) => {
                run.advance_to(fix.t, self.gyro_z, &self.tuning.q)?;
                let z = run.origin.to_enu(fix);
                run.correct(&z, &gps_h(&run.x), &gps_jacobian(), &gps_r(&self.tuning.gps))
            }
        }
    }
}
