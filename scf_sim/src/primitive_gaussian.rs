use std::f64::consts::PI;

use scf_core::point::Point;

use crate::utils::double_factorial;

/// An unnormalized primitive gaussian function
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PrimitiveGaussian {
    n: u8,
    l: u8,
    m: u8,
    alpha: f64,
    center: Point,
}

impl PrimitiveGaussian {
    pub(crate) fn new(n: u8, l: u8, m: u8, alpha: f64, center: Point) -> Self {
        // TODO: check these assertions
        assert!(n < 7); // no element exists with orbitals with principal number higher than 7
        assert!(l <= n);
        assert!(m < 7);
        assert!(alpha > 0.0);
        PrimitiveGaussian {
            n,
            l,
            m,
            alpha,
            center,
        }
    }

    pub(crate) fn alpha(&self) -> f64 {
        self.alpha
    }

    pub(crate) fn center(&self) -> &Point {
        &self.center
    }

    pub fn normalization_constant(&self) -> f64 {
        normalization_coefficient(self.n, self.alpha)
            * normalization_coefficient(self.l, self.alpha)
            * normalization_coefficient(self.m, self.alpha)
    }

    pub fn evaluate(&self, r: &Point) -> f64 {
        let r_a = r.sub(&self.center);
        r_a.x.powi(self.n.into())
            * r_a.y.powi(self.l.into())
            * r_a.z.powi(self.m.into())
            * (-self.alpha * r_a.norm_squared()).exp()
    }
}

fn normalization_coefficient(k: u8, alpha: f64) -> f64 {
    ((2.0 * alpha) / PI).powf(0.25)
        * (4.0 * alpha).powf((k as f64) / 2.0)
        * (double_factorial((2 * k - 1) as i32) as f64).powf(-0.5)
}
