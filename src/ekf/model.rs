use nalgebra::{SMatrix, SVector};
use std::ops::{Index, IndexMut};

pub type Vec5 = SVector<f64, 5>;
pub type Mat5 = SMatrix<f64, 5,5>;

pub fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    (a + PI).rem_euclid(TAU) - PI
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum StateEnum {
    Pe,
    Pn,
    Psi,
    V,
    Bw
}

impl StateEnum {
    pub const DIM: usize = 5;
    pub const ALL: [StateEnum; Self::DIM] = [StateEnum::Pe, StateEnum::Pn, 
        StateEnum::Psi, StateEnum::V, StateEnum::Bw];

    #[inline]
    pub const fn ix(self) -> usize { self as usize }
}

impl Index<StateEnum> for Vec5 {
    type Output = f64;
    #[inline]
    fn index(&self, i: StateEnum) -> &f64 { &self[i.ix()] }
}

impl IndexMut<StateEnum> for Vec5 {
    #[inline]
    fn index_mut(&mut self, i: StateEnum) -> &mut f64 { &mut self[i.ix()] }
}

pub trait StateMatrix {
    fn at(&self, r: StateEnum, c: StateEnum) -> f64;
    fn set(&mut self, r: StateEnum, c: StateEnum, v:f64);
}

impl StateMatrix for Mat5 {
    #[inline]
    fn at(&self, r: StateEnum, c: StateEnum) -> f64 { self[(r.ix(), c.ix())] }
    #[inline]
    fn set(&mut self, r: StateEnum, c: StateEnum, v: f64) { self[(r.ix(), c.ix())] = v; }
}

// Model time
//

#[derive(Debug, Clone, Copy)]
pub struct ProcessNoise {
    pub sigma_omega: f64,
    pub sigma_v: f64,
    pub sigma_b: f64
}

impl Default for ProcessNoise {
    fn default() -> Self {
        Self {
            sigma_omega: 2.0e-3,
            sigma_v: 5.0e-1,
            sigma_b: 1.0e-4
        }
    }
}

pub fn propagate(x: &Vec5, omega_z: f64, dt: f64) -> Vec5 {
    debug_assert!(dt > 0.0, "dt must be a positive number, got {dt}");

    let (psi, v, b) = (x[StateEnum::Psi], x[StateEnum::V], x[StateEnum::Bw]);
    let (s, c) = psi.sin_cos();

    let mut out = *x;
    out[StateEnum::Pe] += v * c * dt;
    out[StateEnum::Pn] += v * s * dt;
    out[StateEnum::Psi] = wrap_pi(psi + (omega_z - b) * dt);

    out
}

pub fn jacobian(x: &Vec5, dt: f64) -> Mat5 {
    let (s,c) = x[StateEnum::Psi].sin_cos();
    let v = x[StateEnum::V];

    let mut f = Mat5::identity();
    f.set(StateEnum::Pe, StateEnum::Psi, -v * s * dt);
    f.set(StateEnum::Pe, StateEnum::V, c*dt);
    f.set(StateEnum::Pn, StateEnum::Psi, v * c * dt);
    f.set(StateEnum::Pn, StateEnum::V, s * dt);
    f.set(StateEnum::Psi, StateEnum::Bw, -dt);
    f
}

pub fn predict(
    x: &Vec5,
    p: &Mat5,
    omega_z: f64,
    dt: f64,
    q: &ProcessNoise
) -> (Vec5, Mat5) {
    let f = jacobian(x, dt);
    let mut qd = Mat5::zeros();
    qd.set(StateEnum::Psi, StateEnum::Psi, q.sigma_omega.powi(2) * dt);
    qd.set(StateEnum::V, StateEnum::V, q.sigma_v.powi(2) * dt);
    qd.set(StateEnum::Bw, StateEnum::Bw, q.sigma_b.powi(2) * dt);

    (propagate(x, omega_z, dt), f * p * f.transpose() + qd)
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    #[test]
    fn enum_is_contiguous() {
        for (n,s) in StateEnum::ALL.iter().enumerate() {
            assert_eq!(s.ix(), n);
        }
    }

    #[test]
    fn jacobian_matches_finite_diff() {
        let x = Vec5::new(10.0, -5.0, 0.7, 12.0, 0.01);
        let (omega, dt, h) = (0.15, 0.02, 1e-6);

        let f = jacobian(&x, dt);

        for col in StateEnum::ALL {
            let (mut hi, mut lo) = (x,x);
            hi[col] += h;
            lo[col] -= h;
            let fd = (propagate(&hi, omega, dt) - propagate(&lo, omega, dt)) / (2.0 * h);

            for row in StateEnum::ALL {
                assert_relative_eq!(f.at(row, col), fd[row], epsilon=1e-6);
            }
        }
    }

    #[test]
    fn straight_line_is_v_t() {
        let b = 0.01;
        let x0 = Vec5::new(0.0, 0.0, 0.0, 10.0, b);
        let (dt, n) = (0.01, 500);

        let mut x = x0;
        for _ in 0..n {
            x = propagate(&x, b, dt)
        }

        assert_relative_eq!(x[StateEnum::Pe], 10.0 * dt * n as f64, epsilon=1e-9);
        assert_relative_eq!(x[StateEnum::Pn], 0.0, epsilon=1e-12);
        assert_relative_eq!(x[StateEnum::Psi], 0.0, epsilon=1e-12);
        assert_relative_eq!(x[StateEnum::V], 10.0, epsilon=1e-12);
        assert_relative_eq!(x[StateEnum::Bw], b, epsilon=1e-12);
    }

    #[test]
    fn check_pi2_rotation_from_east_to_north() {
        let x = Vec5::new(0.0, 0.0, FRAC_PI_2, 10.0, 0.0);
        let out = propagate(&x, 0.0, 0.1);
        assert_relative_eq!(out[StateEnum::Pe], 0.0, epsilon=1e-12);
        assert_relative_eq!(out[StateEnum::Pn], 1.0, epsilon=1e-12);
    }

    #[test]
    fn predict_yields_symmetric_covariance() {
        let x = Vec5::new(1.0, 2.0, 0.4, 8.0, 2e-3);
        let p = Mat5::from_diagonal(&Vec5::new(4.0, 4.0, 0.1, 1.0, 1e-4));
        let (_, pp) = predict(&x, &p, 0.1, 0.01, &ProcessNoise::default());

        assert_relative_eq!(pp, pp.transpose(), epsilon=1e-12);
        assert!(pp.at(StateEnum::Pe, StateEnum::Pe) > p.at(StateEnum::Pe, StateEnum::Pe));
    }

    #[test]
    fn wrap_pi_is_periodic() {
        for &a in &[0.0, 0.3, -0.3, 3.0, -3.0, FRAC_PI_2, -FRAC_PI_2] {
            assert_relative_eq!(wrap_pi(a), a, epsilon=1e-12);
            assert_relative_eq!(wrap_pi(a + TAU), a, epsilon=1e-12);
            assert_relative_eq!(wrap_pi(a - TAU), a, epsilon=1e-12);
            assert_relative_eq!(wrap_pi(wrap_pi(a)), a, epsilon=1e-12);
        }
        assert_relative_eq!(wrap_pi(-PI - 0.1), PI - 0.1, epsilon=1e-12);
    }
}
