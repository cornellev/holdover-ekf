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
    use approx::{assert_relative_eq, assert_relative_ne};

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

    #[test]
    fn equal_p_and_k_splits_diff() {
        // Scalar Kalman gain: P = R => K = 1/2, so estimate will be halfway,
        // and the variance halves as well.
        let (x, p) = prior();
        let z = Vec2::new(2.0, 0.0);
        let r = Mat2::identity() * 4.0;
        let (xn, pn) = update(&x, &p, &z, &gps_h(&x), &gps_jacobian(), &r).unwrap();

        assert_relative_eq!(xn[S::Pe], 1.0, epsilon=1e-12);
        assert_relative_eq!(pn.at(S::Pe, S::Pe), 2.0, epsilon=1e-12);
    }

    #[test]
    fn update_stays_psd_symmetric() {
        let (x, p) = prior();
        let (x, p) = predict(&x, &p, 0.2, 0.1, &ProcessNoise::default());
        let z = Vec2::new(1.3, -0.4);
        let (_, pn) = update(&x, &p, &z, &gps_h(&x), &gps_jacobian(), &gps_r(&GPSNoise::default())).unwrap();

        assert_relative_eq!(pn, pn.transpose(), epsilon=1e-12);
        let min_eig = pn.symmetric_eigenvalues().min();
        assert!(min_eig >= 0.0, "minimum eigenvalue {min_eig}");
    }

    #[test]
    fn position_fix_corrects_heading_through_cross_cov() {
        let (x, p) = prior();
        let (x, p) = predict(&x, &p, 0.0, 0.1, &ProcessNoise::default());
        assert!(p.at(S::Pn, S::Psi) > 0.0);

        let z = gps_h(&x) + Vec2::new(0.0, 1.0);
        let (xn, _) = update(&x, &p, &z, &gps_h(&x), &gps_jacobian(), &gps_r(&GPSNoise::default())).unwrap();

        assert!(xn[S::Psi] > x[S::Psi]);
    }

    #[test]
    fn singular_innovation_returns_none() {
        let x = Vec5::zeros();
        let p = Mat5::zeros();
        let r = Mat2::zeros();
        assert!(update(&x, &p, &Vec2::zeros(), &gps_h(&x), &gps_jacobian(), &r).is_none());
    }
}
