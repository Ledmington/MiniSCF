pub(crate) fn double_factorial(mut n: i32) -> i32 {
    assert!(n >= -1);
    if n <= 1 {
        return 1;
    }
    let mut s = 1;
    while n > 1 {
        s *= n;
        n -= 2;
    }
    s
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::utils::double_factorial;

    #[rstest]
    #[case(0, 1)]
    #[case(1, 1)]
    #[case(2, 2)]
    #[case(3, 3)]
    #[case(4, 8)]
    #[case(5, 15)]
    #[case(6, 48)]
    #[case(7, 105)]
    fn test_double_factorial(#[case] input: i32, #[case] expected: i32) {
        assert_eq!(expected, double_factorial(input));
    }
}
