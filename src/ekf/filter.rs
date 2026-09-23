use nalgebra::{SMatrix, SVector};

use crate::ekf::model::{wrap_pi, Mat5, S, Vec5};

pub fn update<const M: usize>(
    x: &Vec5,
    p: &Mat5,
    z: &SVector<f64, M>,
    z_hat: &SVector<f64, M>,
    h: &SMatrix<f64, M, 5>,
    r: &SMatrix<f64, M, M>,
) -> Option<(Vec5, Mat5)> {
    let y = z - z_hat;
    let s = h * p * h.transpose() + r;
    let s_inv = s.try_inverse()?;
    let k =p * h.transpose() * s_inv;

    let mut x_new = x + k * y;
    x_new[S::Psi] = wrap_pi(x_new[S::Psi]);

    let i_kh = Mat5::identity() - k * h;
    let p_new = i_kh * p * i_kh.transpose() + k * r * k.transpose();

    Some((x_new, p_new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ekf::model::{predict, ProcessNoise, StateMatrix};
    use crate::ekf::sensors::{gps_h, gps_jacobian, gps_r, GPSNoise, Mat2, Vec2};
    use approx::assert_relative_eq;

    fn prior() -> (Vec5, Mat5) {
        let x = Vec5::new(0.0, 0.0, 0.0, 10.0, 0.0);
        let p = Mat5::from_diagonal(&Vec5::new(4.0, 4.0, 0.1, 1.0, 1e-4));
        (x, p)
    }

    #[test]
    fn perfect_measurement_shrinks_cov() {
        let (x,p) = prior();
        let z = gps_h(&x);
        let (xn, pn) = update(&x, &p, &z, &gps_h(&x), &gps_jacobian(), &gps_r(&GPSNoise::default()))
            .expect("S should be invertible");

        assert_relative_eq!(xn, x, epsilon=1e-12);
        assert!(pn.at(S::Pe, S::Pe) < p.at(S::Pe, S::Pe));
        assert!(pn.trace() < p.trace());
    }
}
