
use curve25519_dalek::scalar::Scalar as BigScalar;
use std::fmt::Debug;

pub fn to_scalar(val: i32) -> BigScalar {
    if val < 0 {
        -BigScalar::from((-val) as u32)
    } else {
        BigScalar::from(val as u32)
    }
}

pub fn slice_to_scalar(data: &[i32]) -> Vec<BigScalar> {
    let mut mem: Vec<BigScalar> = Vec::new();
    for &val in data.iter() {
        mem.push(to_scalar(val));
    }
    mem
}

pub fn from_hex_char(hex_str: u8) -> u8{
    match hex_str {
        b'0' => 0, b'1' => 1, b'2' => 2, b'3' => 3, b'4' => 4, b'5' => 5, b'6' => 6, b'7' => 7, b'8' => 8, b'9' => 9, b'a' => 10, b'b' => 11, b'c' => 12, b'd' => 13, b'e' => 14, b'f' => 15, _ => 0
    }
}

pub fn from_hex<T:Scalar>(hex_str: &str) -> T {
    let data = hex_str.as_bytes();
    let len = data.len();
    let mut res: [u8; 32] = [0; 32];
    for i in 0..(len-1)/2 {
        res[i] = from_hex_char(data[len - 2*i - 2]) * 16 + from_hex_char(data[len - 2*i - 1]);
    }
    T::from_bytes(res)
}

pub trait Scalar: Debug + std::ops::Mul<Self, Output=Self> + std::ops::MulAssign<Self> + std::ops::Add<Self, Output=Self> + std::ops::AddAssign<Self> + std::ops::Sub<Self, Output=Self> + std::ops::Neg<Output=Self> + std::cmp::Eq + Clone + Copy {
    fn one() -> Self;
    fn zero() -> Self;
    fn from_i32(x: i32) -> Self;
    fn is_nonneg(&self) -> bool;
    fn to_bytes(&self) -> Vec<u8>;
    fn to_big_scalar(x: &[Self]) -> Vec<BigScalar>;
    fn slice_u32_to_scalar(x: &[u32]) -> Self;
    fn to_i32(&self) -> i32;
    fn from_bytes(data: [u8;32]) -> Self;
    fn invert(&self) -> Self;
}

fn pack_four_bytes(x : &[u8]) -> u32 {
    return (x[0] as u32) | (x[1] as u32) << 8 | (x[2] as u32) << 16 | (x[3] as u32) << 24;
}

pub fn scalar_to_vec_u32(x: BigScalar) -> [u32;8]{
    let t = x.as_bytes();
    [pack_four_bytes(&t[0..4]),
    pack_four_bytes(&t[4..8]),
    pack_four_bytes(&t[8..12]),
    pack_four_bytes(&t[12..16]),
    pack_four_bytes(&t[16..20]),
    pack_four_bytes(&t[20..24]),
    pack_four_bytes(&t[24..28]),
    pack_four_bytes(&t[28..32])]
}

pub fn to_vec_i32<T: Scalar>(x: &[T]) -> Vec<i32> {
    let mut res = Vec::new();
    for &i in x {
        let val = i.to_i32();
        res.push(val);
    }
    res
}

pub fn power_of_two<T : Scalar>(x: u32) -> T {
    let mut t = [0u8;32];
    t[(x/8) as usize] = 1u8 << (x%8);
    T::from_bytes(t)
}

pub const SCALAR_SIZE: u32 = 252;

impl Scalar for i32 {
    fn one() -> i32 {1}
    fn zero() -> i32 {0}
    fn from_i32(x: i32) -> i32 { x }
    fn is_nonneg(&self) -> bool {*self >= 0}
    fn to_bytes(&self) -> Vec<u8> {
        let mut res: Vec<u8> = Vec::with_capacity(4);
        let mut x = *self;
        for _ in 0..4 {
            res.push((x & 255) as u8);
            x >>= 8;
        }
        res
    }
    fn to_big_scalar(x: &[i32]) -> Vec<BigScalar> {
        return slice_to_scalar(x);
    }
    fn slice_u32_to_scalar(x: &[u32]) -> i32 {
        let tmp = BigScalar::slice_u32_to_scalar(x);
        if tmp.is_nonneg() {
            x[0] as i32
        } else {
            -((-tmp).as_bytes()[0] as i32)
        }
    }
    fn to_i32(&self) -> i32 {
        *self
    }
    fn from_bytes(data: [u8;32]) -> Self {
        return pack_four_bytes(&data[0..4]) as i32;
    }
    fn invert(&self) -> Self {
        panic!("not implemented");
    }
}

impl Scalar for BigScalar {
    fn one() -> BigScalar {BigScalar::one()}
    fn zero() -> BigScalar {BigScalar::zero()}
    fn from_i32(x: i32) -> BigScalar { if x < 0 {-BigScalar::from((-x) as u32)} else {BigScalar::from(x as u32)}}
    fn is_nonneg(&self) -> bool { (self.as_bytes()[31] >> 4) == 0 }
    fn to_bytes(&self) -> Vec<u8> {self.to_bytes().to_vec()}
    fn to_big_scalar(x: &[BigScalar]) -> Vec<BigScalar> {
        return x.to_vec();
    }
    fn slice_u32_to_scalar(x: &[u32]) -> BigScalar {
        BigScalar::from_bits([(x[0] & 255) as u8, ((x[0] >> 8) & 255) as u8, ((x[0] >> 16) & 255) as u8, ((x[0] >>24) & 255) as u8,
                                    (x[1] & 255) as u8, ((x[1] >> 8) & 255) as u8, ((x[1] >> 16) & 255) as u8, ((x[1] >>24) & 255) as u8,
                                    (x[2] & 255) as u8, ((x[2] >> 8) & 255) as u8, ((x[2] >> 16) & 255) as u8, ((x[2] >>24) & 255) as u8,
                                    (x[3] & 255) as u8, ((x[3] >> 8) & 255) as u8, ((x[3] >> 16) & 255) as u8, ((x[3] >>24) & 255) as u8,
                                    (x[4] & 255) as u8, ((x[4] >> 8) & 255) as u8, ((x[4] >> 16) & 255) as u8, ((x[4] >>24) & 255) as u8,
                                    (x[5] & 255) as u8, ((x[5] >> 8) & 255) as u8, ((x[5] >> 16) & 255) as u8, ((x[5] >>24) & 255) as u8,
                                    (x[6] & 255) as u8, ((x[6] >> 8) & 255) as u8, ((x[6] >> 16) & 255) as u8, ((x[6] >>24) & 255) as u8,
                                    (x[7] & 255) as u8, ((x[7] >> 8) & 255) as u8, ((x[7] >> 16) & 255) as u8, ((x[7] >>24) & 255) as u8])
    }
    fn to_i32(&self) -> i32 {
        if self.is_nonneg() {
            let val = self.as_bytes();
            pack_four_bytes(&val[0..4]) as i32
        } else {
            let val = -self;
            let val = val.as_bytes();
            -(pack_four_bytes(&val[0..4]) as i32)
        }
    }
    fn from_bytes(data: [u8;32]) -> Self {
        return BigScalar::from_bits(data);
    }
    fn invert(&self) -> Self {
        assert_ne!(self, &Self::zero());
        self.invert()
    }
}
