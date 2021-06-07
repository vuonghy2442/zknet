
use crate::{r1cs::{ConstraintSystem, Functions, Id, Memory, MemoryManager, Range, RangeFrom, RangeTo, TensorAddress, ScalarAddress}, tensor::VariableTensor};
use curve25519_dalek::scalar::Scalar as BigScalar;
use crate::scalar::{from_hex, Scalar, slice_to_scalar};

mod constant;

fn split_temp_tensor(x: &VariableTensor) -> [VariableTensor; 3] {
    // (constant::T*2) * constant::R_F + constant::R_P* 2 + (constant::R_F + constant::R_P - 1) * constant::T
    let temp_f = x.at(&[RangeTo(..(constant::T*2 * constant::R_F) as u32)]).reshape(&[constant::R_F as u32, (constant::T*2) as u32]);
    let temp_p = x.at(&[Range((constant::T*2 * constant::R_F) as u32..(constant::T*2 * constant::R_F + constant::R_P *2) as u32)]).reshape(&[constant::R_P as u32, 2]);
    let round_output = x.at(&[Range((constant::T*2 * constant::R_F + constant::R_P *2) as u32..constant::TEMP_SIZE as u32)]).reshape(&[(constant::R_F + constant::R_P - 1) as u32, constant::T as u32]);
    [temp_f, temp_p, round_output]
}


impl ConstraintSystem {
    fn run_added_poseidon_perm_box<T: Scalar>(input: &[ScalarAddress], output: &[ScalarAddress], temp: [VariableTensor; 3], var_dict: &mut Memory<T>) {
        let [temp_f, temp_p, round_output] = temp;
        // temp constant::T * 2
        fn full_round_x5<T: Scalar>(x: &[ScalarAddress], out: &[ScalarAddress], temp: &[ScalarAddress], round_constant: Option<usize>, var_dict: &mut Memory<T>) {
            // x^5
            let mut x5: [T; constant::T] = [T::zero(); constant::T];
            for i in 0..constant::T {
                let x = var_dict[x[i] as usize];
                let x2 = x*x;
                let x4 = x2*x2;
                x5[i] = x*x4;
                var_dict[temp[2*i] as usize] =  x2;
                var_dict[temp[2*i + 1] as usize] = x4;
            }

            for i in 0..constant::T {
                let mut sum = T::zero();
                for j in 0..constant::T {
                    sum += from_hex::<T>(constant::MDS[i][j]) * x5[j];
                }
                if let Some(rc) = round_constant {
                    sum += from_hex::<T>(constant::ROUND_CONST[rc + i]);
                }
                var_dict[out[i] as usize] = sum;
            }
        }
        // temp : len 2
        fn partial_round_x5<T: Scalar>(x: &[ScalarAddress], out: &[ScalarAddress], temp: &[ScalarAddress], round_constant: Option<usize>, var_dict: &mut Memory<T>) {
            // x^5
            let x0 = var_dict[x[0] as usize];
            let x0_2 = x0*x0;
            let x0_4 = x0_2*x0_2;
            let x0_5 = x0_4*x0;
            var_dict[temp[0] as usize] = x0_2;
            var_dict[temp[1] as usize] = x0_4;

            let mut val: [T; constant::T] = [T::zero(); constant::T];
            val[0] = x0_5;
            for i in 1..constant::T {
                val[i] = var_dict[x[i] as usize];
            }

            for i in 0..constant::T {
                let mut sum = T::zero();
                for j in 0..constant::T {
                    sum += from_hex::<T>(constant::MDS[i][j]) * val[j];
                }
                if let Some(rc) = round_constant {
                    sum += from_hex::<T>(constant::ROUND_CONST[rc + i]);
                }
                var_dict[out[i] as usize] = sum;
            }
        }

        let r_f = constant::R_F / 2;
        // addding constant
        let mut cur_state= input.to_vec();
        for i in 0..r_f {
            let next_state = round_output.at_(&[i as u32]).iter().collect::<Vec<u32>>();
            full_round_x5( &cur_state, &next_state,
                        &temp_f.at_(&[i as u32]).iter().collect::<Vec<u32>>(),
                         Some((i+1)*constant::T), var_dict);
            cur_state = next_state;
        }
        // middle rounds
        for i in r_f..r_f+constant::R_P {
            let next_state = round_output.at_(&[i as u32]).iter().collect::<Vec<u32>>();
            partial_round_x5( &cur_state, &next_state,
                        &temp_p.at_(&[(i - r_f) as u32]).iter().collect::<Vec<u32>>(),
                        Some((i+1)*constant::T), var_dict);
            cur_state = next_state;
        }
        // final rounds - 1
        for i in r_f+constant::R_P..constant::R_F+constant::R_P - 1 {
            let next_state = round_output.at_(&[i as u32]).iter().collect::<Vec<u32>>();
            full_round_x5( &cur_state, &next_state,
                        &temp_f.at_(&[(i - constant::R_P) as u32]).iter().collect::<Vec<u32>>(),
                        Some((i+1)*constant::T), var_dict);
            cur_state = next_state;
        }
        full_round_x5( &cur_state, output,
            &temp_f.at_(&[(constant::R_F - 1) as u32]).iter().collect::<Vec<u32>>(),
            None, var_dict);

    }

    // it doesn't have add round key for round 1 yet so you have to add it manually
    // temp is 1d array of size (constant::T*3) * constant::R_F + constant::R_P* (T + 2) - T
    fn added_poseidon_perm_box(&mut self, input: &[ScalarAddress], output: &[ScalarAddress], temp: VariableTensor)  {
        // temp constant::T * 2
        fn full_round_x5(c: &mut ConstraintSystem, x: &[ScalarAddress], out: &[ScalarAddress], temp: &[ScalarAddress], round_constant: Option<usize> ) {
            // x^5
            for i in 0..constant::T {
                c.a.push((c.n_cons, x[i], BigScalar::one()));
                c.b.push((c.n_cons, x[i], BigScalar::one()));
                c.c.push((c.n_cons, temp[2*i], BigScalar::one()));
                c.n_cons += 1;

                c.a.push((c.n_cons, temp[2*i], BigScalar::one()));
                c.b.push((c.n_cons, temp[2*i], BigScalar::one()));
                c.c.push((c.n_cons, temp[2*i+1], BigScalar::one()));
                c.n_cons += 1;

                c.a.push((c.n_cons, x[i], BigScalar::one()));
                c.b.push((c.n_cons, temp[2*i+1], BigScalar::one()));
                let mut r_const = BigScalar::zero();
                for k in 0..constant::T {
                    let mul = from_hex::<BigScalar>(constant::INVERSE_MDS[i][k]);
                    c.c.push((c.n_cons, out[k], mul));
                    if let Some(r) = round_constant {
                        r_const += mul * from_hex::<BigScalar>(constant::ROUND_CONST[r + k]);
                    }
                }
                if r_const != BigScalar::zero() {
                    c.c.push((c.n_cons, c.mem.one_var, -r_const));
                }
                c.n_cons += 1;

            }
        }
        // temp : len 2
        fn partial_round_x5(c: &mut ConstraintSystem, x: &[ScalarAddress], out: &[ScalarAddress], temp: &[ScalarAddress], round_constant: Option<usize> ) {
            // x^5
            let x2: ScalarAddress = temp[0];
            let x4: ScalarAddress = temp[1];
            c.a.push((c.n_cons, x[0], BigScalar::one()));
            c.b.push((c.n_cons, x[0], BigScalar::one()));
            c.c.push((c.n_cons, x2, BigScalar::one()));
            c.n_cons += 1;

            c.a.push((c.n_cons, x2, BigScalar::one()));
            c.b.push((c.n_cons, x2, BigScalar::one()));
            c.c.push((c.n_cons, x4, BigScalar::one()));
            c.n_cons += 1;

            c.a.push((c.n_cons, x[0], BigScalar::one()));
            c.b.push((c.n_cons, x4, BigScalar::one()));
            for i in 1..constant::T {
                c.a.push((c.n_cons + i as u32, x[i], BigScalar::one()));
                c.b.push((c.n_cons + i as u32, c.mem.one_var, BigScalar::one()));
            }

            for i in 0..constant::T {
                let mut r_const = BigScalar::zero();
                for k in 0..constant::T {
                    let mul = from_hex::<BigScalar>(constant::INVERSE_MDS[i][k]);
                    c.c.push((c.n_cons, out[k], mul));
                    if let Some(r) = round_constant {
                        r_const += mul * from_hex::<BigScalar>(constant::ROUND_CONST[r + k]);
                    }
                }
                if r_const != BigScalar::zero() {
                    c.c.push((c.n_cons, c.mem.one_var, -r_const));
                }
                c.n_cons += 1;
            }
        }

        let input_size = input.len() as u32;
        assert_eq!(input_size, constant::T as u32);
        let r_f = constant::R_F / 2;

        let [temp_f, temp_p, round_output] = split_temp_tensor(&temp);

        // addding constant
        let mut cur_state= input.to_vec();
        for i in 0..r_f {
            let next_state = round_output.at_(&[i as u32]).iter().collect::<Vec<u32>>();
            full_round_x5(self, &cur_state, &next_state,
                        &temp_f.at_(&[i as u32]).iter().collect::<Vec<u32>>(),
                        Some((i+1)*constant::T));
            cur_state = next_state;
        }
        // middle rounds
        for i in r_f..r_f+constant::R_P {
            let next_state = round_output.at_(&[i as u32]).iter().collect::<Vec<u32>>();
            partial_round_x5(self, &cur_state, &next_state,
                        &temp_p.at_(&[(i - r_f) as u32]).iter().collect::<Vec<u32>>(),
                        Some((i+1)*constant::T));
            cur_state = next_state;
        }
        // final rounds - 1
        for i in r_f+constant::R_P..constant::R_F+constant::R_P - 1 {
            let next_state = round_output.at_(&[i as u32]).iter().collect::<Vec<u32>>();
            full_round_x5(self, &cur_state, &next_state,
                        &temp_f.at_(&[(i - constant::R_P) as u32]).iter().collect::<Vec<u32>>(),
                         Some((i+1)*constant::T));
            cur_state = next_state;
        }
        full_round_x5(self, &cur_state, output,
            &temp_f.at_(&[(constant::R_F - 1) as u32]).iter().collect::<Vec<u32>>(),
            None);
    }

    pub fn run_poseidon_perm_box<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        if let [input,output, input_added, temp] = *param {
            for (i, (y,x)) in mem[input_added].iter().zip(mem[input].iter()).enumerate() {
                var_dict[y as usize] = var_dict[x as usize] + from_hex(constant::ROUND_CONST[i]);
            }
            Self::run_added_poseidon_perm_box(
                &mem[input_added].iter().collect::<Vec<u32>>(),
                &mem[output].iter().collect::<Vec<u32>>(),
                split_temp_tensor(&mem[temp]), var_dict
            );
        } else {
            panic!("Params doesn't match");
        }

    }

    fn poseidon_perm_box(&mut self, input: TensorAddress, output: TensorAddress) {
        let input_added = self.mem.alloc(&[constant::T as u32]); //add round key
        for (i, (y,x)) in self.mem[input_added].iter().zip(self.mem[input].iter()).enumerate() {
            self.a.push((self.n_cons, x, BigScalar::one()));
            self.a.push((self.n_cons, self.mem.one_var, from_hex(constant::ROUND_CONST[i])));
            self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
            self.c.push((self.n_cons, y, BigScalar::one()));
            self.n_cons += 1;
        }
        let temp = self.mem.alloc(&[constant::TEMP_SIZE as u32]);
        self.added_poseidon_perm_box(
            &self.mem[input_added].iter().collect::<Vec<u32>>(),
            &self.mem[output].iter().collect::<Vec<u32>>(),
            self.mem[temp].clone()
        );

        self.compute.push((Box::new([input, output, input_added, temp]), Functions::PoseidonPerm))
    }

    pub fn poseidon_hash(&mut self, input: &[TensorAddress], output: TensorAddress) {
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_poseidon_perm() {
        let mut c = ConstraintSystem::new();
        let data = c.mem.alloc(&[3]);
        let output = c.mem.alloc(&[3]);
        c.poseidon_perm_box(data, output);

        println!("Constraints {}", c.n_cons);

        let mut mem = c.mem.new_memory::<BigScalar>();
        c.load_memory(data, &mut mem, &slice_to_scalar(&[1,2,3]));
        c.sort_cons();
        c.compute(&mut mem);

        assert!(c.verify(&mem));

        assert_eq!(&mem[c.mem[output].begin() as usize..c.mem[output].end() as usize],[
            BigScalar::from_bits([215, 17, 46, 105, 231, 118, 107, 215, 151, 251, 29, 117, 91, 152, 43, 125, 30, 245, 98, 158, 249, 160, 96, 247, 242, 177, 110, 173, 136, 202, 249, 5]),
            BigScalar::from_bits([44, 91, 7, 19, 225, 230, 152, 101, 103, 253, 203, 97, 123, 183, 146, 174, 17, 84, 73, 72, 40, 114, 12, 208, 249, 107, 65, 202, 79, 229, 186, 11]),
            BigScalar::from_bits([201, 113, 137, 17, 45, 182, 4, 132, 109, 198, 198, 194, 150, 245, 140, 156, 217, 196, 175, 214, 147, 230, 55, 186, 204, 150, 220, 14, 211, 15, 63, 11])
        ]);


    }
}