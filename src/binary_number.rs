use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign};

type BinNumInner = u8;

/// A wrapper around an integer limited to value 0 and 1.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub struct BinNum {
    pub(crate) inner: BinNumInner,
}

impl BinNum {
    /// Returns a new BinNum.
    ///
    /// # Panic
    ///
    /// Panics if the given number is not 0 or 1.
    pub fn new(number: BinNumInner) -> Self {
        if !(number == 0 || number == 1) {
            panic!("binary number must be 0 or 1")
        }
        BinNum { inner: number }
    }

    /// Returns the binary number 1.
    pub fn one() -> Self {
        BinNum { inner: 1 }
    }

    /// Returns the binary number 0.
    pub fn zero() -> Self {
        BinNum { inner: 0 }
    }

    /// Checks if a binary number has value 1.
    pub fn is_one(self) -> bool {
        self.inner == 1
    }

    /// Checks if a binary number has value 0.
    pub fn is_zero(self) -> bool {
        self.inner == 0
    }
}

impl From<BinNumInner> for BinNum {
    fn from(number: BinNumInner) -> Self {
        Self::new(number)
    }
}

impl From<BinNum> for BinNumInner {
    fn from(number: BinNum) -> Self {
        number.inner
    }
}

impl Add<BinNum> for BinNum {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        BinNum {
            inner: self.inner ^ other.inner,
        }
    }
}

impl AddAssign<BinNum> for BinNum {
    fn add_assign(&mut self, other: Self) {
        self.inner ^= other.inner;
    }
}

impl Mul<BinNum> for BinNum {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        BinNum {
            inner: self.inner * other.inner,
        }
    }
}

impl MulAssign<BinNum> for BinNum {
    fn mul_assign(&mut self, other: Self) {
        self.inner *= other.inner;
    }
}

impl fmt::Display for BinNum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}
