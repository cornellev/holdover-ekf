use nalgebra::{SMatrix, SVector};
use sensor_defs::{GPS, IMU, Reject};

use crate::ekf::filter::update;
use crate::ekf::model::{predict, Mat5, ProcessNoise, Vec5};
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


#[derive(Debug)]
struct Running {
    origin: Origin,
    x: Vec5,
    p: Mat5,
    t_ns: u64,
}

impl Running {
    // Heading frmo the chord between two fixes; variance from sigma_gps at each end.
    fn from_two_fixes(origin: Origin, first: Vec2, t0_ns: u64, z: Vec2, t_ns: u64, tun: &Tuning) -> Self {
        let d = z - first;
        let (len, dt) = (d.norm(), (t_ns - t0_ns) as f64 * 1e-9);
        let var_gps = tun.gps.sigma_pos.powi(2);

        let x = Vec5::new(z[0], z[1], d[1].atan2(d[0]), len / dt, 0.0);
        let p = Mat5::from_diagonal(&Vec5::new(
            var_gps,
            var_gps,
            2.0 * var_gps / len.powi(2),
            2.0 * var_gps / dt.powi(2),
            tun.sigma_b0.powi(2)
        ));

        Self { origin, x, p, t_ns }
    }

    fn advance_to(&mut self, t_ns: u64, gyro_z: f64, q: &ProcessNoise) -> Result<(), Reject> {
        // Order check must come before the u64 subtraction, or it underflows.
        if t_ns < self.t_ns {
            return Err(Reject::OutOfOrder { stamp_ns: t_ns, last_ns: self.t_ns });
        }
        if t_ns > self.t_ns {
            let dt = (t_ns - self.t_ns) as f64 * 1e-9;
            (self.x, self.p) = predict(&self.x, &self.p, gyro_z, dt, q);
            self.t_ns = t_ns;
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
    AwaitingMotion { origin: Origin, first: Vec2, t0_ns: u64 },
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
            self.gyro_z = imu.gyro_rad_s()[2];
            return Err(Reject::NotRunning);
        };

        // Hold the *previous* gyro smaple over [t_prev, imu.t] then latch onto the new one when ready
        run.advance_to(imu.stamp_ns(), self.gyro_z, &self.tuning.q)?;
        self.gyro_z = imu.gyro_rad_s()[2];

        run.correct(
            &Vec1::new(imu.accel_m_s2()[1]),
            &imu_h(&run.x, self.gyro_z),
            &imu_jacobian(&run.x, self.gyro_z),
            &imu_r(&self.tuning.imu),
        )
    }

    pub fn on_gps(&mut self, fix: &GPS) -> Result<(), Reject> {
        match &mut self.phase {
            Phase::AwaitingFix => {
                self.phase = Phase::AwaitingMotion {
                    origin: Origin::from_fix(fix),
                    first: Vec2::zeros(),
                    t0_ns: fix.stamp_ns(),
                };
                Ok(())
            }
            Phase::AwaitingMotion { origin, first, t0_ns} => {
                let z = origin.to_enu(fix);
                if (z - *first).norm() >= self.tuning.min_baseline && fix.stamp_ns() > *t0_ns {
                    let run = Running::from_two_fixes(*origin, *first, *t0_ns, z, fix.stamp_ns(), &self.tuning);
                    self.phase = Phase::Running(run);
                }
                Ok(())
            }
            Phase::Running(run) => {
                run.advance_to(fix.stamp_ns(), self.gyro_z, &self.tuning.q)?;
                let z = run.origin.to_enu(fix);
                run.correct(&z, &gps_h(&run.x), &gps_jacobian(), &gps_r(&self.tuning.gps))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    const S: u64 = 1_000_000_000;
    const LAT: f64 = 37.4;
    const LON: f64 = -122.1;

    fn fix(stamp_ns: u64, dlat_deg: f64) -> Result<GPS, Box<dyn Error>> {
        Ok(GPS::new(stamp_ns, LAT + dlat_deg, LON, 30.0, [6.25; 3])?)
    }

    fn running_ekf() -> Result<EKF, Box<dyn Error>> {
        let mut ekf = EKF::new(Tuning::default());
        ekf.on_gps(&fix(0, 0.0)?)?;
        ekf.on_gps(&fix(S, 2e-4)?)?;
        assert!(ekf.estimate().is_some(), "setup should reaach running");
        Ok(ekf)
    }

    #[test]
    fn stale_imu_is_rejected() -> Result<(), Box<dyn Error>> {
        let mut ekf = running_ekf()?;
        let (x, p) = ekf.estimate().expect("running");
        let (x0, p0) = (*x, *p);

        let stale = IMU::new(S / 2, [0.0; 3], [0.0; 3])?;
        let err = ekf.on_imu(&stale).unwrap_err();
        assert_eq!(err, Reject::OutOfOrder { stamp_ns: S / 2, last_ns: S });
        
        let (x, p) = ekf.estimate().expect("still running");
        assert_eq!((*x, *p), (x0, p0));
        Ok(())
    }

    #[test]
    fn stale_second_fix_does_not_start_filter() -> Result<(), Box<dyn Error>> {
        let mut ekf = EKF::new(Tuning::default());
        ekf.on_gps(&fix(5 * S, 0.0)?)?;
        ekf.on_gps(&fix(2 * S, 2e-4)?)?;
        assert!(ekf.estimate().is_none());
        Ok(())
    }
}
