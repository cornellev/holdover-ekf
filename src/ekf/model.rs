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

