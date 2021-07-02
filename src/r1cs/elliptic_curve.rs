use std::{convert::TryInto, usize};

use crate::{r1cs::Functions, tensor::VariableTensor};

use super::{BigScalar, ConstraintSystem, ScalarAddress, Scalar, Memory, MemoryManager, scalar_to_vec_u32};

// we use a selected an Edwards curve x^2 + y^2 = 1 + 1408 x^2 y^2 (mod p) where p = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed
// this curve is selected so that it satisfies the requirements in http://safecurves.cr.yp.to/index.html and has small cofactor
// this curve has order of 2^2 * 1809251394333065553493296640760748560224405988559295410502714166441017894153
const ORDER_BIT: u32 = 251;

// reduced order
pub fn get_order<T: Scalar>() -> T{
    const ORDER: [u8; 32] = [9, 101, 184, 12, 125, 201, 154, 3, 25, 91, 234, 114, 31, 29, 214, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4];
    T::from_bytes(ORDER)
}

pub fn get_a<T: Scalar>() -> T {
    T::one()
}

pub fn get_d<T: Scalar>() -> T {
    T::from_i32(1408)
}

pub fn get_id<T: Scalar>() -> [T; 2] {
    [T::zero(), T::one()]
}

pub fn elliptic_add<T: Scalar>(a: &[T], b: &[T], param_a: T, param_d: T) -> [T; 2] {
    let (t0,t1,t2,t3) = (a[0] * b[1],a[1]*b[0], a[1]*b[1], a[0]*b[0] * param_a);
    let t4 = t0 * t1 * param_d;
    [
        (t0 + t1) * (T::one() + t4).invert(),
        (t2 - t3) * (T::one() - t4).invert()
    ]
}

pub fn elliptic_mul<T: Scalar>(a: &[T], scalar: T, param_a: T, param_d: T) -> [T; 2] {
    let bytes = scalar.to_bytes();
    let mut base: [T;2] = a.try_into().unwrap();
    let mut sum = [T::zero(), T::one()];
    for i in 0..ORDER_BIT {
        let bit = (bytes[(i / 8) as usize] >> i % 8) & 1;
        if bit == 1{
            sum = elliptic_add(&sum, &base, param_a, param_d);
        }
        base = elliptic_add(&base, &base, param_a, param_d);
    };
    sum
}

impl ConstraintSystem {
    fn run_elliptic_add<T: Scalar>(a: &[ScalarAddress], b: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress], param_a: T, param_d: T,  var_dict: &mut Memory<T>) {
        // temp[0] = a[0] * b[1]
        var_dict[temp[0] as usize] = var_dict[a[0] as usize] * var_dict[b[1] as usize];
        // temp[1] = a[1] * b[0]
        var_dict[temp[1] as usize] = var_dict[a[1] as usize] * var_dict[b[0] as usize];
        // temp[2] = a[1] * b[1]
        var_dict[temp[2] as usize] = var_dict[a[1] as usize] * var_dict[b[1] as usize];
        // temp[3] = param_a*a[0] * b[0]
        var_dict[temp[3] as usize] = var_dict[a[0] as usize] * var_dict[b[0] as usize] * param_a;
        // temp[4] = temp[0] * temp[1] * d
        var_dict[temp[4] as usize] = var_dict[temp[0] as usize] * var_dict[temp[1] as usize] * param_d;
        // res[0] = (temp[0] + temp[1])/(1+temp[4])
        var_dict[res[0] as usize] = (var_dict[temp[0] as usize] + var_dict[temp[1] as usize]) * (T::one() + var_dict[temp[4] as usize]).invert();
        // res[1] = (temp[2] - temp[3])/(1-temp[4])
        var_dict[res[1] as usize] = (var_dict[temp[2] as usize] - var_dict[temp[3] as usize]) * (T::one() - var_dict[temp[4] as usize]).invert();
    }

    // temp with size of five
    fn elliptic_add(&mut self, a: &[ScalarAddress], b: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress], param_a: BigScalar, param_d: BigScalar) {
        // temp[0] = a[0] * b[1]
        self.a.push((self.n_cons, a[0], BigScalar::one()));
        self.b.push((self.n_cons, b[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[0], BigScalar::one()));
        self.n_cons += 1;
        // temp[1] = a[1] * b[0]
        self.a.push((self.n_cons, a[1], BigScalar::one()));
        self.b.push((self.n_cons, b[0], BigScalar::one()));
        self.c.push((self.n_cons, temp[1], BigScalar::one()));
        self.n_cons += 1;
        // temp[2] = a[1] * b[1]
        self.a.push((self.n_cons, a[1], BigScalar::one()));
        self.b.push((self.n_cons, b[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[2], BigScalar::one()));
        self.n_cons += 1;
        // temp[3] = param_a*a[0] * b[0]
        self.a.push((self.n_cons, a[0], param_a));
        self.b.push((self.n_cons, b[0], BigScalar::one()));
        self.c.push((self.n_cons, temp[3], BigScalar::one()));
        self.n_cons += 1;
        // temp[4] = temp[0] * temp[1] * d
        self.a.push((self.n_cons, temp[0], param_d));
        self.b.push((self.n_cons, temp[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[4], BigScalar::one()));
        self.n_cons += 1;
        // res[0] = (temp[0] + temp[1])/(1+temp[4])
        self.a.push((self.n_cons, res[0], BigScalar::one()));
        self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
        self.b.push((self.n_cons, temp[4], BigScalar::one()));
        self.c.push((self.n_cons, temp[0], BigScalar::one()));
        self.c.push((self.n_cons, temp[1], BigScalar::one()));
        self.n_cons += 1;
        // res[1] = (temp[2] - temp[3])/(1-temp[4])
        self.a.push((self.n_cons, res[1], BigScalar::one()));
        self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
        self.b.push((self.n_cons, temp[4], -BigScalar::one()));
        self.c.push((self.n_cons, temp[2], BigScalar::one()));
        self.c.push((self.n_cons, temp[3], -BigScalar::one()));
        self.n_cons += 1;
    }

    fn run_elliptic_double<T: Scalar>(a: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress], param_a: T,  var_dict: &mut Memory<T>) {
        // temp[0] = a[0] * a[1]
        var_dict[temp[0] as usize] = var_dict[a[0] as usize] * var_dict[a[1] as usize];
        // temp[1] = param_a * a[0] * a[0]
        var_dict[temp[1] as usize] = var_dict[a[0] as usize] * var_dict[a[0] as usize] * param_a;
        // temp[2] = a[1] * a[1]
        var_dict[temp[2] as usize] = var_dict[a[1] as usize] * var_dict[a[1] as usize];
        // res[0] = 2 * temp[0] / (temp[1] + temp[2])
        let sum = var_dict[temp[1] as usize] + var_dict[temp[2] as usize];
        var_dict[res[0] as usize] = T::from_i32(2) * var_dict[temp[0] as usize] * sum.invert();
        // res[1] = (temp[1] - temp[2]) / (temp[1] + temp[2] - 2)
        var_dict[res[1] as usize] = (var_dict[temp[1] as usize] - var_dict[temp[2] as usize]) * (sum - T::from_i32(2)).invert();
    }

    // temp should have size of two
    fn elliptic_double(&mut self, a: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress], param_a: BigScalar) {
        // temp[0] = a[0] * a[1]
        self.a.push((self.n_cons, a[0], BigScalar::one()));
        self.b.push((self.n_cons, a[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[0], BigScalar::one()));
        self.n_cons += 1;
        // temp[1] = param_a * a[0] * a[0]
        self.a.push((self.n_cons, a[0], param_a));
        self.b.push((self.n_cons, a[0], BigScalar::one()));
        self.c.push((self.n_cons, temp[1], BigScalar::one()));
        self.n_cons += 1;
        // temp[2] = a[1] * a[1]
        self.a.push((self.n_cons, a[1], BigScalar::one()));
        self.b.push((self.n_cons, a[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[2], BigScalar::one()));
        self.n_cons += 1;
        // res[0] = 2 * temp[0] / (temp[1] + temp[2])
        self.a.push((self.n_cons, res[0], BigScalar::one()));
        self.b.push((self.n_cons, temp[1], BigScalar::one()));
        self.b.push((self.n_cons, temp[2], BigScalar::one()));
        self.c.push((self.n_cons, temp[0], BigScalar::from_i32(2)));
        self.n_cons += 1;
        // res[1] = (temp[1] - temp[2]) / (temp[1] + temp[2] - 2)
        self.a.push((self.n_cons, res[1], BigScalar::one()));
        self.b.push((self.n_cons, temp[1], BigScalar::one()));
        self.b.push((self.n_cons, temp[2], BigScalar::one()));
        self.b.push((self.n_cons, self.mem.one_var, BigScalar::from_i32(-2)));
        self.c.push((self.n_cons, temp[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[2], -BigScalar::one()));
        self.n_cons += 1;
    }

    pub(super) fn run_elliptic_mul<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let param_a = T::slice_u32_to_scalar(&param[..8]);
        let param_d = T::slice_u32_to_scalar(&param[8..16]);

        if let [bits, tmp, tmp_last, a0, a1, res0, res1] = param[16..] {

            // zero_var
            let zero_var = mem[tmp_last].at_idx(&[7]);
            var_dict[zero_var as usize] = T::zero();

            let mut sum: Vec<ScalarAddress> = vec![zero_var, mem.one_var];
            let mut base: Vec<ScalarAddress> = vec![a0, a1];

            for (i, bit) in mem[bits].iter().enumerate() {
                let last = (i as u32) == ORDER_BIT - 1;
                let tmp = if last {
                    mem[tmp_last].clone()
                } else {
                    mem[tmp].at_(&[i as u32])
                }.to_vec();

                let res_add = &tmp[5..7];
                Self::run_elliptic_add(&sum, &base, &tmp[0..5], res_add, param_a, param_d, var_dict);

                let new_sum = if last {
                    [res0, res1]
                } else {
                    tmp[7..9].try_into().unwrap()
                };

                for i in 0..2 {
                    var_dict[new_sum[i] as usize] = var_dict[if var_dict[bit as usize] == T::zero() {
                        sum[i]
                    } else {
                        res_add[i]
                    } as usize];
                }


                if !last {
                    let res_double = &tmp[12..14];
                    Self::run_elliptic_double(&base, &tmp[9..12], res_double, param_a, var_dict);
                    base = res_double.to_vec();
                    sum = new_sum.to_vec();
                }
            }
        } else {
            panic!("params don't match")
        }
    }

    pub fn elliptic_mul(&mut self, a: &[ScalarAddress], scalar: ScalarAddress, res: &[ScalarAddress], param_a: BigScalar, param_d: BigScalar) {
        let scalar = self.mem.save(VariableTensor::new_const(scalar, &[1]));
        let bits = self.mem.alloc(&[1, ORDER_BIT]);
        let sign = self.mem.alloc(&[1]);
        let two_complement = self.mem.alloc(&[1]);
        self.bit_decomposition(scalar, bits, sign, two_complement, true);

        let tmp = self.mem.alloc(&[ORDER_BIT, 14]);
        let tmp_last = self.mem.alloc(&[8]);

        // zero_var
        let zero_var = self.mem[tmp_last].at_idx(&[7]);
        self.c.push((self.n_cons, zero_var, BigScalar::one()));
        self.n_cons += 1;

        let mut sum: Vec<ScalarAddress> = vec![zero_var, self.mem.one_var];
        let mut base: Vec<ScalarAddress> = a.to_vec();

        for (i, bit) in self.mem[bits].iter().enumerate() {
            let last = (i as u32) == ORDER_BIT - 1;
            let tmp = if last {
                self.mem[tmp_last].clone()
            } else {
                self.mem[tmp].at_(&[i as u32])
            }.to_vec();

            let res_add = &tmp[5..7];
            self.elliptic_add(&sum, &base, &tmp[0..5], res_add, param_a, param_d);

            let new_sum = if last {
                &res
            } else {
                &tmp[7..9]
            };

            for i in 0..2 {
                self.a.push((self.n_cons, bit, BigScalar::one()));
                self.b.push((self.n_cons, sum[i], -BigScalar::one()));
                self.b.push((self.n_cons, res_add[i], BigScalar::one()));
                self.c.push((self.n_cons, new_sum[i], BigScalar::one()));
                self.c.push((self.n_cons, sum[i], -BigScalar::one()));
                self.n_cons += 1;
            }

            if !last {
                let res_double = &tmp[12..14];
                self.elliptic_double(&base, &tmp[9..12], res_double, param_a);
                base = res_double.to_vec();
                sum = new_sum.to_vec();
            }
        }

        let mut params: Vec<u32> = Vec::new();
        params.extend_from_slice(&scalar_to_vec_u32(param_a));
        params.extend_from_slice(&scalar_to_vec_u32(param_d));
        params.extend([bits, tmp, tmp_last, a[0], a[1], res[0], res[1]]);
        self.compute.push((params.into_boxed_slice(), Functions::EllipticMul));
    }

    pub(super) fn run_elliptic_add_cond<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let param_a = T::slice_u32_to_scalar(&param[..8]);
        let param_d = T::slice_u32_to_scalar(&param[8..16]);
        if let [tmp, cond, a0, a1, b0, b1, res0, res1] = param[16..] {
            let tmp = &mem[tmp].to_vec();
            let res = [res0, res1];
            let a = [a0, a1];
            let b = [b0, b1];
            Self::run_elliptic_add(&a, &b, &tmp[..5], &tmp[5..7], param_a, param_d, var_dict);

            for i in 0..2 {
                var_dict[res[i] as usize] = var_dict[if var_dict[cond as usize] == T::zero() {
                    a[i]
                } else {
                    tmp[5+i]
                } as usize];
            }

        } else {
            panic!("params don't match");
        }
    }

    // return a + b if cond == 1 otherwise return a
    pub fn elliptic_add_cond(&mut self, a: &[ScalarAddress], b: &[ScalarAddress], res: &[ScalarAddress], cond: ScalarAddress, param_a: BigScalar, param_d: BigScalar) {
        let tmp = self.mem.alloc(&[7]);
        let tmp_tensor = self.mem[tmp].to_vec();
        let sum = &tmp_tensor[5..7];
        self.elliptic_add(a, b, &tmp_tensor[0..5], &sum, param_a, param_d);

        for i in 0..2 {
            self.a.push((self.n_cons, cond, BigScalar::one()));
            self.b.push((self.n_cons, a[i], -BigScalar::one()));
            self.b.push((self.n_cons, sum[i], BigScalar::one()));
            self.c.push((self.n_cons, res[i], BigScalar::one()));
            self.c.push((self.n_cons, a[i], -BigScalar::one()));
            self.n_cons += 1;
        }

        let mut params: Vec<u32> = Vec::new();
        params.extend_from_slice(&scalar_to_vec_u32(param_a));
        params.extend_from_slice(&scalar_to_vec_u32(param_d));
        params.extend([tmp, cond, a[0], a[1], b[0], b[1], res[0], res[1]]);
        self.compute.push((params.into_boxed_slice(), Functions::EllipticAddCond));
    }
}

#[cfg(test)]
mod test {
    use super::{ConstraintSystem, get_a, get_d, BigScalar};
    use crate::{r1cs::elliptic_curve::get_order, scalar::Scalar};
    fn get_p() -> [BigScalar; 2] {
        [
            BigScalar::from(4u8),
            BigScalar::from_bits([76, 130, 153, 225, 227, 248, 189, 252, 230, 93, 83, 140, 31, 24, 25, 166, 160, 41, 87, 122, 154, 76, 139, 222, 19, 171, 116, 205, 184, 155, 121, 5] )
        ]
    }

    #[test]
    fn elliptic_mul_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2]);
        let scalar = x.mem.alloc(&[1]);
        let output = x.mem.alloc(&[2]);
        x.elliptic_mul(&[1,2], 3, &[4,5], get_a(), get_d());
        x.reorder_for_spartan(&[input]);
        let mut mem = x.mem.new_memory();
        println!("Number of constraints {}", x.cons_size());
        x.load_memory(input, &mut mem, &get_p());
        x.load_memory(scalar, &mut mem, &[get_order()]);

        x.compute(&mut mem);

        assert_eq!(mem[1], BigScalar::from_bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert_eq!(mem[2], BigScalar::from_bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));


        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn elliptic_add_cond_test_2() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2]);

        let _output = x.mem.alloc(&[2]);
        let cond = x.mem.alloc(&[1]);
        x.elliptic_add_cond(&[1,2], &[1,2], &[3,4], x.mem[cond].begin(), get_a(), get_d());
        x.reorder_for_spartan(&[input]);
        let mut mem = x.mem.new_memory();
        println!("Number of constraints {}", x.cons_size());
        x.load_memory(input, &mut mem,  &[BigScalar::from_bytes([165, 69, 15, 204, 207, 113, 207, 38, 62, 63, 78, 98, 124, 5, 127, 19, 227, 172, 104, 57, 76, 114, 16, 216, 22, 108, 66, 159, 246, 205, 84, 4]), BigScalar::from_bytes([117, 164, 25, 122, 240, 125, 11, 247, 5, 194, 218, 37, 43, 92, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]) ]);
        x.load_memory(cond, &mut mem, &[BigScalar::from_i32(0)]);

        x.compute(&mut mem);
        assert_eq!(mem[0], mem[x.mem[input].to_vec()[0] as usize]);
        assert_eq!(mem[1], mem[x.mem[input].to_vec()[1] as usize]);

        x.sort_cons();
        assert!(x.verify(&mem));
    }
}