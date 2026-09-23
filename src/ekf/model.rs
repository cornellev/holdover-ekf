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
pub enum S {
    Pe,
    Pn,
    Psi,
    V,
    Bw
}

impl S {
    pub const DIM: usize = 5;
    pub const ALL: [S; Self::DIM] = [S::Pe, S::Pn, 
        S::Psi, S::V, S::Bw];

    #[inline]
    pub const fn ix(self) -> usize { self as usize }
}

impl Index<S> for Vec5 {
    type Output = f64;
    #[inline]
    fn index(&self, i: S) -> &f64 { &self[i.ix()] }
}

impl IndexMut<S> for Vec5 {
    #[inline]
    fn index_mut(&mut self, i: S) -> &mut f64 { &mut self[i.ix()] }
}

pub trait StateMatrix {
    fn at(&self, r: S, c: S) -> f64;
    fn set(&mut self, r: S, c: S, v:f64);
}

impl StateMatrix for Mat5 {
    #[inline]
    fn at(&self, r: S, c: S) -> f64 { self[(r.ix(), c.ix())] }
    #[inline]
    fn set(&mut self, r: S, c: S, v: f64) { self[(r.ix(), c.ix())] = v; }
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

    let (psi, v, b) = (x[S::Psi], x[S::V], x[S::Bw]);
    let (s, c) = psi.sin_cos();

    let mut out = *x;
    out[S::Pe] += v * c * dt;
    out[S::Pn] += v * s * dt;
    out[S::Psi] = wrap_pi(psi + (omega_z - b) * dt);

    out
}

pub fn jacobian(x: &Vec5, dt: f64) -> Mat5 {
    let (s,c) = x[S::Psi].sin_cos();
    let v = x[S::V];

    let mut f = Mat5::identity();
    f.set(S::Pe, S::Psi, -v * s * dt);
    f.set(S::Pe, S::V, c*dt);
    f.set(S::Pn, S::Psi, v * c * dt);
    f.set(S::Pn, S::V, s * dt);
    f.set(S::Psi, S::Bw, -dt);
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
    qd.set(S::Psi, S::Psi, q.sigma_omega.powi(2) * dt);
    qd.set(S::V, S::V, q.sigma_v.powi(2) * dt);
    qd.set(S::Bw, S::Bw, q.sigma_b.powi(2) * dt);

    (propagate(x, omega_z, dt), f * p * f.transpose() + qd)
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    #[test]
    fn enum_is_contiguous() {
        for (n,s) in S::ALL.iter().enumerate() {
            assert_eq!(s.ix(), n);
        }
    }

    #[test]
    fn jacobian_matches_finite_diff() {
        let x = Vec5::new(10.0, -5.0, 0.7, 12.0, 0.01);
        let (omega, dt, h) = (0.15, 0.02, 1e-6);

        let f = jacobian(&x, dt);

        for col in S::ALL {
            let (mut hi, mut lo) = (x,x);
            hi[col] += h;
            lo[col] -= h;
            let fd = (propagate(&hi, omega, dt) - propagate(&lo, omega, dt)) / (2.0 * h);

            for row in S::ALL {
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

        assert_relative_eq!(x[S::Pe], 10.0 * dt * n as f64, epsilon=1e-9);
        assert_relative_eq!(x[S::Pn], 0.0, epsilon=1e-12);
        assert_relative_eq!(x[S::Psi], 0.0, epsilon=1e-12);
        assert_relative_eq!(x[S::V], 10.0, epsilon=1e-12);
        assert_relative_eq!(x[S::Bw], b, epsilon=1e-12);
    }

    #[test]
    fn check_pi2_rotation_from_east_to_north() {
        let x = Vec5::new(0.0, 0.0, FRAC_PI_2, 10.0, 0.0);
        let out = propagate(&x, 0.0, 0.1);
        assert_relative_eq!(out[S::Pe], 0.0, epsilon=1e-12);
        assert_relative_eq!(out[S::Pn], 1.0, epsilon=1e-12);
    }

    #[test]
    fn predict_yields_symmetric_covariance() {
        let x = Vec5::new(1.0, 2.0, 0.4, 8.0, 2e-3);
        let p = Mat5::from_diagonal(&Vec5::new(4.0, 4.0, 0.1, 1.0, 1e-4));
        let (_, pp) = predict(&x, &p, 0.1, 0.01, &ProcessNoise::default());

        assert_relative_eq!(pp, pp.transpose(), epsilon=1e-12);
        assert!(pp.at(S::Pe, S::Pe) > p.at(S::Pe, S::Pe));
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
