use std::usize;

use crate::{r1cs::Functions, tensor::VariableTensor};

use super::{BigScalar, ConstraintSystem, ScalarAddress, Scalar, Memory, MemoryManager, scalar_to_vec_u32};

// we use a selected curve of y^2 = x^3 + 16x + 289 (mod p) where p = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed
// this curve have a montgomery form y^2 = x^3 + M x^2 + x where M = 4755651596668048722525231845714201167747526728278866922525026204640692435799
// this curve is selected so that it satisfies the requirements in http://safecurves.cr.yp.to/index.html and has small cofactor
// this curve has order of 2^3 * 904625697166532776746648320380374280092004119092104366837763295296274715673
const ORDER_BIT: u32 = 249;

pub fn get_m<T: Scalar>() -> T {
    T::from_bytes([87, 123, 68, 2, 238, 149, 233, 82, 35, 161, 81, 88, 28, 174, 212, 151, 98, 29, 169, 139, 111, 74, 111, 121, 241, 103, 100, 135, 121, 154, 131, 10])
}

impl ConstraintSystem {
    fn run_elliptic_add<T: Scalar>(a: &[ScalarAddress], b: &[ScalarAddress], diff: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress],  var_dict: &mut Memory<T>) {
        var_dict[temp[0] as usize] = (var_dict[a[0] as usize] - var_dict[a[1] as usize]) * (var_dict[b[0] as usize] + var_dict[b[1] as usize]);
        var_dict[temp[1] as usize] = (var_dict[a[0] as usize] + var_dict[a[1] as usize]) * (var_dict[b[0] as usize] - var_dict[b[1] as usize]);
        let t = var_dict[temp[0] as usize] + var_dict[temp[1] as usize];
        var_dict[res[0] as usize] = t * t;

        let t = var_dict[temp[0] as usize] - var_dict[temp[1] as usize];
        var_dict[temp[2] as usize] = t * t;
        var_dict[res[1] as usize] = var_dict[temp[2] as usize] * var_dict[diff[0] as usize];
    }

    // temp with size of three
    fn elliptic_add(&mut self, a: &[ScalarAddress], b: &[ScalarAddress], diff: ScalarAddress, temp: &[ScalarAddress], res: &[ScalarAddress]) {
        // tmp[0] = (a[0] - a[1]) * (b[0] + b[1])
        self.a.push((self.n_cons, a[0], BigScalar::one()));
        self.a.push((self.n_cons, a[1], -BigScalar::one()));
        self.b.push((self.n_cons, b[0], BigScalar::one()));
        self.b.push((self.n_cons, b[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[0], BigScalar::one()));
        self.n_cons += 1;

        // tmp[1] = (a[0] + a[1]) * (b[0] - b[1])
        self.a.push((self.n_cons, a[0], BigScalar::one()));
        self.a.push((self.n_cons, a[1], BigScalar::one()));
        self.b.push((self.n_cons, b[0], BigScalar::one()));
        self.b.push((self.n_cons, b[1], -BigScalar::one()));
        self.c.push((self.n_cons, temp[1], BigScalar::one()));
        self.n_cons += 1;

        // res[0] == (temp[0] + temp[1])^2
        self.a.push((self.n_cons, temp[0], BigScalar::one()));
        self.a.push((self.n_cons, temp[1], BigScalar::one()));
        self.b.push((self.n_cons, temp[0], BigScalar::one()));
        self.b.push((self.n_cons, temp[1], BigScalar::one()));
        self.c.push((self.n_cons, res[0], BigScalar::one()));
        self.n_cons += 1;

        // temp[2] == (temp[0] - temp[1])^2
        self.a.push((self.n_cons, temp[0], BigScalar::one()));
        self.a.push((self.n_cons, temp[1], -BigScalar::one()));
        self.b.push((self.n_cons, temp[0], BigScalar::one()));
        self.b.push((self.n_cons, temp[1], -BigScalar::one()));
        self.c.push((self.n_cons, temp[2], BigScalar::one()));
        self.n_cons += 1;

        // res[1] = diff[0] * temp[2]
        self.a.push((self.n_cons, temp[2], BigScalar::one()));
        self.b.push((self.n_cons, diff, BigScalar::one()));
        self.c.push((self.n_cons, res[1], BigScalar::one()));
        self.n_cons += 1;
    }

    fn run_elliptic_double<T: Scalar>(a: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress], m: T,  var_dict: &mut Memory<T>) {
        // temp[0] = (a[0] + a[1])^2
        let t = var_dict[a[0] as usize] + var_dict[a[1] as usize];
        var_dict[temp[0] as usize] = t * t;
        // temp[1] = (a[0] - a[1])^2
        let t = var_dict[a[0] as usize] - var_dict[a[1] as usize];
        var_dict[temp[1] as usize] = t * t;

        // res[0] = temp[0] * temp[1]
        var_dict[res[0] as usize] = var_dict[temp[0] as usize] * var_dict[temp[1] as usize];

        // res[1] = (temp[0] - temp[1]) * (temp[1] + (M + 2)/4 * (temp[0] - temp[1]))
        let t =  var_dict[temp[0] as usize] - var_dict[temp[1] as usize];
        var_dict[res[1] as usize] = t * (var_dict[temp[1] as usize] + (m + T::from_i32(2))*T::from_i32(4).invert() * t);
    }

    // temp should have size of two
    fn elliptic_double(&mut self, a: &[ScalarAddress], temp: &[ScalarAddress], res: &[ScalarAddress], m: BigScalar) {
        // temp[0] = (a[0] + a[1])^2
        self.a.push((self.n_cons, a[0], BigScalar::one()));
        self.a.push((self.n_cons, a[1], BigScalar::one()));
        self.b.push((self.n_cons, a[0], BigScalar::one()));
        self.b.push((self.n_cons, a[1], BigScalar::one()));
        self.c.push((self.n_cons, temp[0], BigScalar::one()));
        self.n_cons += 1;
        // temp[1] = (a[0] - a[1])^2
        self.a.push((self.n_cons, a[0], BigScalar::one()));
        self.a.push((self.n_cons, a[1], -BigScalar::one()));
        self.b.push((self.n_cons, a[0], BigScalar::one()));
        self.b.push((self.n_cons, a[1], -BigScalar::one()));
        self.c.push((self.n_cons, temp[1], BigScalar::one()));
        self.n_cons += 1;
        // res[0] = temp[0] * temp[1]
        self.a.push((self.n_cons, temp[0], BigScalar::one()));
        self.b.push((self.n_cons, temp[1], BigScalar::one()));
        self.c.push((self.n_cons, res[0], BigScalar::one()));
        self.n_cons += 1;
        // res[1] = (temp[0] - temp[1]) * (temp[1] + (M + 2)/4 * (temp[0] - temp[1]))
        // =  (temp[0] - temp[1]) * ((2 - M)/4 * temp[1] + (M + 2)/4 * temp[0])
        self.a.push((self.n_cons, temp[0], BigScalar::one()));
        self.a.push((self.n_cons, temp[1], -BigScalar::one()));
        self.b.push((self.n_cons, temp[0], (BigScalar::from(2u8) + m)*BigScalar::from(4u8).invert()));
        self.b.push((self.n_cons, temp[1], (BigScalar::from(2u8) - m)*BigScalar::from(4u8).invert()));
        self.c.push((self.n_cons, res[1], BigScalar::one()));
        self.n_cons += 1;
    }

    pub(super) fn run_elliptic_mul<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let m = T::slice_u32_to_scalar(&param[..8]);
        if let [bits, tmp, a, res] = param[8..] {

            // zero_var
            let zero_var = mem[tmp].at_idx(&[0,1]);
            var_dict[zero_var as usize] = T::zero();

            let diff: [ScalarAddress; 2] = [a, mem.one_var];

            let mut p: Vec<ScalarAddress> = vec![mem.one_var, zero_var];
            let mut q: Vec<ScalarAddress> = vec![a, mem.one_var];

            for (i, bit) in mem[bits].reverse(1).iter().enumerate() {
                let tmp = mem[tmp].at_(&[i as u32]).iter().collect::<Vec<u32>>();

                let res_add = if i > 0 {
                    let res_add = &tmp[0..2];
                    Self::run_elliptic_add(&p, &q, &diff, &tmp[2..5], &res_add, var_dict);
                    res_add.to_vec()
                } else {
                    q.clone()
                };

                //double_p 6..8
                //double_Q 10..12
                let double_p = &tmp[7..9];
                let double_q = &tmp[11..13];
                Self::run_elliptic_double(&p,  &tmp[5..7], &double_p, m, var_dict);
                Self::run_elliptic_double(&q,  &tmp[9..11], &double_q, m, var_dict);

                // res_P 12..14
                // res_Q 14..16
                // bit * tmp
                let new_p = tmp[13..15].to_vec();
                let new_q = tmp[15..17].to_vec();
                // new_p = (1 - bit) * add_point + bit * double_p
                // new_q = (1 - bit) * double_q + bit * add_point
                for i in 0..2 {
                    if var_dict[bit as usize] == T::zero() {
                        var_dict[new_p[i] as usize] = var_dict[double_p[i] as usize];
                        var_dict[new_q[i] as usize] = var_dict[res_add[i] as usize];
                    } else {
                        var_dict[new_p[i] as usize] = var_dict[res_add[i] as usize];
                        var_dict[new_q[i] as usize] = var_dict[double_q[i] as usize];
                    }
                }

                p = new_p;
                q = new_q;
            }

            var_dict[res as usize] = var_dict[p[0] as usize] * var_dict[p[1] as usize].invert();
        } else {
            panic!("params don't match")
        }
    }

    pub fn elliptic_mul(&mut self, a: ScalarAddress, scalar: ScalarAddress, res: ScalarAddress, m: BigScalar) {
        let scalar = self.mem.save(VariableTensor::new_const(scalar, &[1]));
        let bits = self.mem.alloc(&[1, ORDER_BIT]);
        let sign = self.mem.alloc(&[1]);
        let two_complement = self.mem.alloc(&[1]);
        self.bit_decomposition(scalar, bits, sign, two_complement, true);

        let tmp = self.mem.alloc(&[ORDER_BIT, 17]);

        // zero_var
        let zero_var = self.mem[tmp].at_idx(&[0,1]);
        self.c.push((self.n_cons, zero_var, BigScalar::one()));
        self.n_cons += 1;

        let mut p: Vec<ScalarAddress> = vec![self.mem.one_var, zero_var];
        let mut q: Vec<ScalarAddress> = vec![a, self.mem.one_var];

        for (i, bit) in self.mem[bits].reverse(1).iter().enumerate() {
            let tmp = self.mem[tmp].at_(&[i as u32]).iter().collect::<Vec<u32>>();

            let res_add = if i > 0 {
                let res_add = &tmp[0..2];
                self.elliptic_add(&p, &q, a, &tmp[2..5], &res_add);
                res_add.to_vec()
            } else {
                q.to_owned()
            };

            //double_p 6..8
            //double_Q 10..12
            let double_p = &tmp[7..9];
            let double_q = &tmp[11..13];
            self.elliptic_double(&p,  &tmp[5..7], &double_p, m);
            self.elliptic_double(&q,  &tmp[9..11], &double_q, m);

            // res_P 12..14
            // res_Q 14..16
            // bit * tmp
            let new_p = tmp[13..15].to_vec();
            let new_q = tmp[15..17].to_vec();
            // new_p = (1 - bit) * add_point + bit * double_p
            // new_q = (1 - bit) * double_q + bit * add_point
            for i in 0..2 {
                self.a.push((self.n_cons, bit, BigScalar::one()));
                self.b.push((self.n_cons, double_p[i], -BigScalar::one()));
                self.b.push((self.n_cons, res_add[i], BigScalar::one()));
                self.c.push((self.n_cons, new_p[i], BigScalar::one()));
                self.c.push((self.n_cons, double_p[i], -BigScalar::one()));
                self.n_cons += 1;

                self.a.push((self.n_cons, bit, BigScalar::one()));
                self.b.push((self.n_cons, res_add[i], -BigScalar::one()));
                self.b.push((self.n_cons, double_q[i], BigScalar::one()));
                self.c.push((self.n_cons, new_q[i], BigScalar::one()));
                self.c.push((self.n_cons, res_add[i], -BigScalar::one()));
                self.n_cons += 1;
            }
            p = new_p;
            q = new_q;
        }

        // final result
        // res = p[0]/p[1]
        self.a.push((self.n_cons, res, BigScalar::one()));
        self.b.push((self.n_cons, p[1], BigScalar::one()));
        self.c.push((self.n_cons, p[0], BigScalar::one()));
        self.n_cons += 1;

        let mut params: Vec<u32> = Vec::new();
        params.extend_from_slice(&scalar_to_vec_u32(m));
        params.extend([bits, tmp, a, res]);

        self.compute.push((params.into_boxed_slice(), Functions::EllipticMul));
    }
}

#[cfg(test)]
mod test {
    use super::{ConstraintSystem, get_m, BigScalar};
    use crate::scalar::{Scalar, slice_to_scalar};
    #[test]
    fn elliptic_mul_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[1]);
        let scalar = x.mem.alloc(&[1]);
        let output = x.mem.alloc(&[1]);
        x.elliptic_mul(x.mem[input].begin(), x.mem[scalar].begin(), x.mem[output].begin(), get_m());
        x.reorder_for_spartan(&[input]);
        let mut mem = x.mem.new_memory();
        x.load_memory(input, &mut mem,  &slice_to_scalar(&[15]));
        x.load_memory(scalar, &mut mem, &slice_to_scalar(&[4172382]));
        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize], BigScalar::from_bytes([152, 47, 223, 157, 227, 139, 208, 148, 2, 16, 252, 104, 235, 119, 218, 230, 77, 92, 187, 45, 81, 241, 255, 149, 155, 239, 10, 94, 196, 84, 11, 13]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
}