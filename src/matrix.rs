use std::{
    fmt::Display,
    ops::{Add, Index, Sub},
};

#[derive(Clone)]
pub(crate) struct Matrix {
    n_rows: usize,
    n_columns: usize,
    data: Vec<f64>,
}

impl Matrix {
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
}

impl Index<usize> for Matrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * self.n_columns)..((index + 1) * self.n_columns)]
    }
}

impl Display for Matrix {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                print!(
                    "{}{:.6}",
                    if self[i][j] < 0.0 { "" } else { "+" },
                    self[i][j]
                );
                if j < self.n_columns - 1 {
                    print!(" ");
                }
            }
            if i < self.n_rows - 1 {
                println!();
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
