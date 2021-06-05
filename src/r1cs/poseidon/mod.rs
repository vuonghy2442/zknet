
use crate::{r1cs::{ConstraintSystem, Functions, Id, Memory, MemoryManager, Range, RangeFrom, TensorAddress}, tensor::VariableTensor};
use curve25519_dalek::scalar::Scalar as BigScalar;
use crate::scalar::{from_hex, Scalar, slice_to_scalar};

mod constant;

impl ConstraintSystem {
    pub fn run_poseidon_perm_box<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        fn add_const<T: Scalar>(input: &VariableTensor, output: &VariableTensor, from: usize, var_dict: &mut Memory<T>) {
            for (i, (old, new)) in input.iter().zip(output.iter()).enumerate() {
                var_dict[new as usize] = var_dict[old as usize] + from_hex::<T>(constant::ROUND_CONST[from + i])
            }
        }
        fn sbox<T: Scalar>(input: &VariableTensor, output: &VariableTensor, temp: &VariableTensor, var_dict: &mut Memory<T>) {
            let mut iter_in = input.iter();
            let mut iter_out = output.iter();

            loop {
                let x = iter_in.next();
                if let None = x { break };
                let x = x.unwrap();
                let y = iter_out.next().unwrap();
                let idx = &iter_in.idx;
                let temp = temp.at_(idx);

                let x2 = temp.at_idx(&[0]);
                let x4 = temp.at_idx(&[1]);
                var_dict[x2 as usize] = var_dict[x as usize] * var_dict[x as usize];
                var_dict[x4 as usize] = var_dict[x2 as usize] * var_dict[x2 as usize];
                var_dict[y as usize] = var_dict[x4 as usize] * var_dict[x as usize];
            }
        }

        fn mds_mixing<T: Scalar>(input: &VariableTensor, output: &VariableTensor, var_dict: &mut Memory<T>) {
            for i in 0..constant::T {
                let mut res = T::zero();
                for j in 0..constant::T {
                    res += var_dict[input.at_idx(&[j as u32]) as usize] * from_hex::<T>(constant::MDS[i][j]);
                }
                var_dict[output.at_idx(&[i as u32]) as usize] = res;
            }
        }

        if let [input, output, temp_sbox_full, temp_sbox_partial, after_add_constant_full, after_add_constant_partial, after_sbox, after_mds] = *param {
            let r_f = constant::R_F / 2;
            let mut state_words: VariableTensor = mem[input].clone();
            // First full rounds
            for r in 0..r_f {
                // Round constants, nonlinear layer, matrix multiplication
                let add_constant_tensor = mem[after_add_constant_full].at_(&[r as u32]);
                add_const( &state_words, &add_constant_tensor, r * constant::T, var_dict);

                let sbox_tensor = mem[after_sbox].at_(&[r as u32]);
                sbox( &add_constant_tensor, &sbox_tensor, &mem[temp_sbox_full].at_(&[r as u32]), var_dict);

                let mds_tensor = mem[after_mds].at_(&[r as u32]);
                mds_mixing( &sbox_tensor, &mds_tensor, var_dict);
                state_words = mds_tensor;
            }


            // Middle partial rounds
            for r in r_f..r_f + constant::R_P {
                // Round constants, nonlinear layer, matrix multiplication
                let add_constant_tensor = mem[after_add_constant_partial].at_(&[(r - r_f) as u32]);
                let sbox_tensor = mem[after_sbox].at(&[Id(r as u32)]);

                add_const( &state_words, &add_constant_tensor, r * constant::T, var_dict);
                add_const( &state_words.at(&[RangeFrom(1..)]), &sbox_tensor.at(&[RangeFrom(1..)]), r * constant::T + 1, var_dict);

                sbox( &add_constant_tensor.at(&[Range(0..1)]), &sbox_tensor.at(&[Range(0..1)]), &mem[temp_sbox_partial].at_(&[(r - r_f) as u32]), var_dict);

                let mds_tensor = mem[after_mds].at_(&[r as u32]);
                mds_mixing( &sbox_tensor, &mds_tensor, var_dict);
                state_words = mds_tensor;
            }

            // Last full rounds
            for r in r_f+constant::R_P..constant::R_F + constant::R_P {
                // Round constants, nonlinear layer, matrix multiplication
                let add_constant_tensor = mem[after_add_constant_full].at_(&[(r - constant::R_P) as u32]);
                add_const( &state_words, &add_constant_tensor, r * constant::T, var_dict);

                let sbox_tensor = mem[after_sbox].at_(&[r as u32]);
                sbox( &add_constant_tensor, &sbox_tensor, &mem[temp_sbox_full].at_(&[(r-constant::R_P) as u32]), var_dict);

                let mds_tensor = if r == constant::R_F + constant::R_P - 1 {
                    mem[output].clone()
                } else {
                    mem[after_mds].at_(&[r as u32])
                };
                mds_mixing( &sbox_tensor, &mds_tensor, var_dict);
                state_words = mds_tensor;
            }
        } else {
            panic!("doesn't match params");
        }
    }
    pub fn poseidon_perm_box(&mut self, input: TensorAddress, output: TensorAddress) {
        fn add_const(c: &mut ConstraintSystem, input: &VariableTensor, output: &VariableTensor, from: usize) {
            for (i,(old, new)) in input.iter().zip(output.iter()).enumerate() {
                c.a.push((c.n_cons, old, BigScalar::one()));
                c.a.push((c.n_cons, c.mem.one_var, from_hex(constant::ROUND_CONST[from + i])));
                c.b.push((c.n_cons, c.mem.one_var, BigScalar::one()));
                c.c.push((c.n_cons, new, BigScalar::one()));
                c.n_cons += 1;
            }
        }
        fn sbox(c: &mut ConstraintSystem, input: &VariableTensor, output: &VariableTensor, temp: &VariableTensor) {
            let mut iter_in = input.iter();
            let mut iter_out = output.iter();

            loop {
                let x = iter_in.next();
                if let None = x { break };
                let x = x.unwrap();
                let y = iter_out.next().unwrap();
                let idx = &iter_in.idx;
                let temp = temp.at_(idx);

                let x2 = temp.at_idx(&[0]);
                let x4 = temp.at_idx(&[1]);
                c.a.push((c.n_cons, x, BigScalar::one()));
                c.b.push((c.n_cons, x, BigScalar::one()));
                c.c.push((c.n_cons, x2, BigScalar::one()));
                c.n_cons += 1;

                c.a.push((c.n_cons, x2, BigScalar::one()));
                c.b.push((c.n_cons, x2, BigScalar::one()));
                c.c.push((c.n_cons, x4, BigScalar::one()));
                c.n_cons += 1;

                c.a.push((c.n_cons, x, BigScalar::one()));
                c.b.push((c.n_cons, x4, BigScalar::one()));
                c.c.push((c.n_cons, y, BigScalar::one()));
                c.n_cons += 1;
            }
        }

        fn mds_mixing(c: &mut ConstraintSystem, input: &VariableTensor, output: &VariableTensor) {
            for i in 0..constant::T {
                for j in 0..constant::T {
                    c.a.push((c.n_cons, input.at_idx(&[j as u32]), from_hex(constant::MDS[i][j])));
                }
                c.b.push((c.n_cons, c.mem.one_var, BigScalar::one()));
                c.c.push((c.n_cons, output.at_idx(&[i as u32]), BigScalar::one()));
                c.n_cons += 1;
            }
        }

        let input_size = self.mem[input].size();
        assert_eq!(input_size, constant::T as u32);
        let r_f = constant::R_F / 2;

        let mut state_words: VariableTensor = self.mem[input].clone();
        let temp_sbox_full = self.mem.alloc(&[constant::R_F as u32, input_size, 2]);
        let temp_sbox_partial = self.mem.alloc(&[constant::R_P as u32, 1, 2]);

        let after_add_constant_full = self.mem.alloc(&[(constant::R_F) as u32, input_size]);
        let after_add_constant_partial = self.mem.alloc(&[(constant::R_P) as u32, 1]);

        let after_sbox = self.mem.alloc(&[(constant::R_F + constant::R_P) as u32, input_size]);
        let after_mds = self.mem.alloc(&[(constant::R_F + constant::R_P - 1) as u32, input_size]);

        // First full rounds
        for r in 0..r_f {
            // Round constants, nonlinear layer, matrix multiplication
            let add_constant_tensor = self.mem[after_add_constant_full].at_(&[r as u32]);
            add_const(self, &state_words, &add_constant_tensor, r * constant::T);

            let sbox_tensor = self.mem[after_sbox].at_(&[r as u32]);
            sbox(self, &add_constant_tensor, &sbox_tensor, &self.mem[temp_sbox_full].at_(&[r as u32]));

            let mds_tensor = self.mem[after_mds].at_(&[r as u32]);
            mds_mixing(self, &sbox_tensor, &mds_tensor);
            state_words = mds_tensor;
        }

        // Middle partial rounds
        for r in r_f..r_f + constant::R_P {
            // Round constants, nonlinear layer, matrix multiplication
            let add_constant_tensor = self.mem[after_add_constant_partial].at_(&[(r - r_f) as u32]);
            let sbox_tensor = self.mem[after_sbox].at(&[Id(r as u32)]);

            add_const(self, &state_words, &add_constant_tensor, r * constant::T);
            add_const(self, &state_words.at(&[RangeFrom(1..)]), &sbox_tensor.at(&[RangeFrom(1..)]), r * constant::T + 1);

            sbox(self, &add_constant_tensor.at(&[Range(0..1)]), &sbox_tensor.at(&[Range(0..1)]), &self.mem[temp_sbox_partial].at_(&[(r - r_f) as u32]));

            let mds_tensor = self.mem[after_mds].at_(&[r as u32]);
            mds_mixing(self, &sbox_tensor, &mds_tensor);
            state_words = mds_tensor;
        }

        // Last full rounds
        for r in r_f+constant::R_P..constant::R_F + constant::R_P {
            // Round constants, nonlinear layer, matrix multiplication
            let add_constant_tensor = self.mem[after_add_constant_full].at_(&[(r - constant::R_P) as u32]);
            add_const(self, &state_words, &add_constant_tensor, r * constant::T);

            let sbox_tensor = self.mem[after_sbox].at_(&[r as u32]);
            sbox(self, &add_constant_tensor, &sbox_tensor, &self.mem[temp_sbox_full].at_(&[(r-constant::R_P) as u32]));

            let mds_tensor = if r == constant::R_F + constant::R_P - 1 {
                self.mem[output].clone()
            } else {
                self.mem[after_mds].at_(&[r as u32])
            };
            mds_mixing(self, &sbox_tensor, &mds_tensor);
            state_words = mds_tensor;
        }
        self.compute.push((Box::new([input, output, temp_sbox_full, temp_sbox_partial, after_add_constant_full, after_add_constant_partial, after_sbox, after_mds]), Functions::PoseidonPerm))
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

        let mut mem = c.mem.new_memory::<BigScalar>();
        c.load_memory(data, &mut mem, &slice_to_scalar(&[1,2,3]));
        c.compute(&mut mem);

        assert!(c.verify(&mem));
        assert_eq!(&mem[c.mem[output].begin() as usize..c.mem[output].end() as usize],[
            BigScalar::from_bits([215, 17, 46, 105, 231, 118, 107, 215, 151, 251, 29, 117, 91, 152, 43, 125, 30, 245, 98, 158, 249, 160, 96, 247, 242, 177, 110, 173, 136, 202, 249, 5]),
            BigScalar::from_bits([44, 91, 7, 19, 225, 230, 152, 101, 103, 253, 203, 97, 123, 183, 146, 174, 17, 84, 73, 72, 40, 114, 12, 208, 249, 107, 65, 202, 79, 229, 186, 11]),
            BigScalar::from_bits([201, 113, 137, 17, 45, 182, 4, 132, 109, 198, 198, 194, 150, 245, 140, 156, 217, 196, 175, 214, 147, 230, 55, 186, 204, 150, 220, 14, 211, 15, 63, 11])
        ]);
    }
}