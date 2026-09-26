use std::time::{Duration, Instant};

use holdover_ekf::ekf::estimator::{EKF, Tuning};
use holdover_ekf::ekf::model::{wrap_pi, Vec5, StateEnum};
use holdover_ekf::ekf::msg::{GPSFix, IMU};
use map_3d::{enu2geodetic, Ellipsoid};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

const LAT0: f64 = 37.4;
const LON0: f64 = -122.1;
const ALT0: f64 = 30.0;

const DT: f64 = 0.01; // 100 Hz IMU
const GPS_RATE: usize = 10; // 10 Hz GPS
const T_END: f64 = 300.0; // seconds
const OUTAGE: (f64, f64) = (150.0, 180.0); // simulation outage of GPS

const V: f64 = 10.0;
const GYRO_BIAS: f64 = 5e-3;

fn yaw_rate(t: f64) -> f64 {
    0.1 * (std::f64::consts::TAU * t / 60.0).sin()
}

fn to_fix(t: f64, e: f64, n: f64) -> GPSFix {
    let (lat, lon, alt) = enu2geodetic(
        e, n, 0.0, 
        LAT0.to_radians(), LON0.to_radians(), ALT0,
        Ellipsoid::WGS84
    );
    GPSFix { t, lat: lat.to_degrees(), lon: lon.to_degrees(), alt }
}

#[derive(Default)]
struct Stats {
    sq_pos: f64,
    n: usize,
    nees: f64,
    n_nees: usize,
}

impl Stats {
    fn rmse(&self) -> f64 {
        (self.sq_pos / self.n.max(1) as f64).sqrt()
    }
    fn mean_nees(&self) -> f64 {
        self.nees / self.n_nees.max(1) as f64
    }
}


fn main() {
    let tuning = Tuning::default();
    let seed = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    let gyro_noise = Normal::new(0.0, tuning.q.sigma_omega / DT.sqrt()).unwrap(); // density 
    // -> per sample
    let ay_noise = Normal::new(0.0, 0.3).unwrap();
    let gps_noise = Normal::new(0.0, tuning.gps.sigma_pos).unwrap();

    let mut ekf = EKF::new(tuning);
    let mut truth = Vec5::new(0.0, 0.0, 0.0, V, GYRO_BIAS);
    // Filter origiin is the first noisy fix to the GPS, so the truth needs to be shifted into that
    // frame.
    let mut origin_offset: Option<(f64, f64)> = None;

    let (mut nominal, mut holdover) = (Stats::default(), Stats::default());
    let mut max_holdover_err: f64 = 0.0;
    let mut filter_time = Duration::ZERO;
    let n_steps = (T_END / DT) as usize;


    println!("{:>6} {:>9} {:>9} {:>9} {:>10} {:>9}", "t[s]", "pos_err", "psi_err", "v_err", "b_w", "sigma_pos");

    for k in 0..n_steps {
        let t = k as f64 * DT;

        // Truth: same kinematics as filter but with exact inputs - no noise.
        let omega = yaw_rate(t);
        if k > 0 {
            let (s, c) = truth[StateEnum::Psi].sin_cos();
            truth[StateEnum::Pe] += V * c * DT;
            truth[StateEnum::Pn] += V * s * DT;
            truth[StateEnum::Psi] = wrap_pi(truth[StateEnum::Psi] + omega * DT);
        }

        let imu = IMU {
            t,
            gyro_z: omega + GYRO_BIAS + gyro_noise.sample(&mut rng),
            accel_x: 0.0,
            accel_y: V * omega + ay_noise.sample(&mut rng),
        };

        let in_outage = (OUTAGE.0..OUTAGE.1).contains(&t);
        let gps = (k % GPS_RATE == 0 && !in_outage).then(|| {
            let (de, dn) = (gps_noise.sample(&mut rng), gps_noise.sample(&mut rng));
            origin_offset.get_or_insert((truth[StateEnum::Pe] + de, truth[StateEnum::Pn] + dn));
            to_fix(t, truth[StateEnum::Pe] + de, truth[StateEnum::Pn] + dn)
        });

        let tic = Instant::now();
        let _ = ekf.on_imu(&imu); // NotRunning until we have a heading
        if let Some(fix) = &gps {
            ekf.on_gps(fix).expect("GPS update failed");
        }
        filter_time += tic.elapsed();

        let (Some((x,p)), Some((oe, on))) = (ekf.estimate(), origin_offset) else {
            continue;
        };


        let mut err = truth - x;
        err[StateEnum::Pe] -= oe;
        err[StateEnum::Pn] -= on;
        err[StateEnum::Psi] = wrap_pi(err[StateEnum::Psi]);
        let pos_err = err.fixed_rows::<2>(0).norm();

        let stats = if in_outage { &mut holdover } else { &mut nominal };
        stats.sq_pos += pos_err.powi(2);
        stats.n += 1;
        if t > 10.0 {
            if let Some(p_inv) = p.try_inverse() {
                stats.nees += (err.transpose() * p_inv * err)[0];
                stats.n_nees += 1;
            }
        }
        if in_outage {
            max_holdover_err = max_holdover_err.max(pos_err);
        }

        if k % 1000 == 0 {
            let sig_pos = (p[(0,0)] + p[(1,1)]).sqrt();
            println!(
            "{t:6.1} {pos_err:9.3} {:9.4} {:9.3} {:10.2e} {sig_pos:9.3}",
                err[StateEnum::Psi], err[StateEnum::V], x[StateEnum::Bw],
            );
        }

    }

    println!("\nnominal : pos RMSE {:.3} m, mean NEES {:.2} (dof 5)", nominal.rmse(), nominal.mean_nees());
    println!(
        "holdover: pos RMSE {:.3} m, max {:.3} m over {:.0} s, mean NEES {:.2}",
        holdover.rmse(), max_holdover_err, OUTAGE.1 - OUTAGE.0, holdover.mean_nees()
    );
    println!("filter : {:.2} us/setep", filter_time.as_secs_f64() * 1e6 / n_steps as f64);

}
