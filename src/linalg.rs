use crate::matrix::Matrix;

/// Factorizes P into U D U^T
pub(crate) fn factorize(p: &Matrix, u: &mut Matrix, d: &mut Matrix) {
    assert!(p.is_square());
    assert_eq!(p.shape(), u.shape());
    assert_eq!(p.shape(), d.shape());
    assert!(p.is_symmetric());
    assert!(u.is_identity());
    assert!(d.is_zero());

    let n = p.rows();

    for j in (0..n).rev() {
        let mut sum_d = 0.0;
        for k in (j + 1)..n {
            sum_d += u[j][k] * u[j][k] * d[k][k];
        }
        d[j][j] = p[j][j] - sum_d;

        if d[j][j].abs() > 1e-12 {
            for i in (0..j).rev() {
                let mut sum_u = 0.0;
                for k in (j + 1)..n {
                    sum_u += u[i][k] * d[k][k] * u[j][k];
                }
                u[i][j] = (p[i][j] - sum_u) / d[j][j];
            }
        }
    }

    assert!(u.is_unit_diagonal() && u.is_upper_triangular());
    assert!(d.is_diagonal());
}
