use std::{
    fmt::Display,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

#[derive(Clone)]
pub(crate) struct Matrix {
    n_rows: usize,
    n_columns: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub(crate) fn zero(n_rows: usize, n_columns: usize) -> Self {
        Matrix {
            n_rows,
            n_columns,
            data: vec![0.0; n_rows * n_columns],
        }
    }

    pub(crate) fn identity(n_rows: usize, n_columns: usize) -> Self {
        let mut m = Matrix::zero(n_rows, n_columns);
        for i in 0..n_rows.min(n_columns) {
            m[i][i] = 1.0;
        }
        m
    }

    pub(crate) fn foreach<X>(elements: &[X], func: fn(&X, &X) -> f64) -> Self {
        let mut data = vec![0.0; elements.len() * elements.len()];
        for i in 0..elements.len() {
            for j in 0..elements.len() {
                data[i * elements.len() + j] = func(&elements[i], &elements[j]);
            }
        }
        Matrix {
            n_rows: elements.len(),
            n_columns: elements.len(),
            data,
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.n_rows
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_columns)
    }

    pub(crate) fn is_square(&self) -> bool {
        self.n_rows == self.n_columns
    }

    pub(crate) fn is_symmetric(&self) -> bool {
        for i in 0..self.n_rows {
            for j in (i + 1)..self.n_columns {
                if self[i][j] != self[j][i] {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn is_diagonal(&self) -> bool {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                if i == j {
                    continue;
                }
                if self[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Checks if the main diagonal is 1
    // TODO: find a better name
    pub(crate) fn is_unit_diagonal(&self) -> bool {
        assert!(self.is_square());
        for i in 0..self.n_rows {
            if self[i][i] != 1.0 {
                return false;
            }
        }
        true
    }

    pub(crate) fn is_identity(&self) -> bool {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                if i == j {
                    if self[i][i] != 1.0 {
                        return false;
                    }
                } else {
                    if self[i][j] != 0.0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(crate) fn is_zero(&self) -> bool {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                if self[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn is_upper_triangular(&self) -> bool {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                if i > j && self[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn transposed(&self) -> Matrix {
        let mut data = vec![0.0; self.data.len()];

        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                data[j * self.n_rows + i] = self[i][j];
            }
        }

        Matrix {
            n_rows: self.n_columns,
            n_columns: self.n_rows,
            data,
        }
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * self.n_columns)..((index + 1) * self.n_columns)]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[(index * self.n_columns)..((index + 1) * self.n_columns)]
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                write!(
                    f,
                    "{}{:.6}",
                    if self[i][j] < 0.0 { "" } else { "+" },
                    self[i][j]
                )?;
                if j < self.n_columns - 1 {
                    write!(f, " ")?;
                }
            }
            if i < self.n_rows - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.n_rows, rhs.n_rows);
        assert_eq!(self.n_columns, rhs.n_columns);
        Matrix {
            n_rows: self.n_rows,
            n_columns: self.n_columns,
            data: (0..(self.n_rows * self.n_columns))
                .map(|i| self.data[i] + rhs.data[i])
                .collect(),
        }
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.n_rows, rhs.n_rows);
        assert_eq!(self.n_columns, rhs.n_columns);
        Matrix {
            n_rows: self.n_rows,
            n_columns: self.n_columns,
            data: (0..(self.n_rows * self.n_columns))
                .map(|i| self.data[i] - rhs.data[i])
                .collect(),
        }
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.n_columns, rhs.n_rows,
            "Cannot multiply matrices: dimensions mismatch ({}x{} and {}x{})",
            self.n_rows, self.n_columns, rhs.n_rows, rhs.n_columns
        );

        let mut result = Matrix::zero(self.n_rows, rhs.n_columns);

        for i in 0..self.n_rows {
            for j in 0..rhs.n_columns {
                let mut sum = 0.0;
                for k in 0..self.n_columns {
                    sum += self[i][k] * rhs[k][j];
                }
                result.data[i * rhs.n_columns + j] = sum;
            }
        }

        result
    }
}
