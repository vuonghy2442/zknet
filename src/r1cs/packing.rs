use super::{ConstraintSystem, BigScalar, MemoryManager, Memory, TensorAddress, Functions};
use crate::{scalar::{Scalar, power_of_two, scalar_to_vec_u32}, tensor::VariableTensor};

impl ConstraintSystem {
    fn run_packing<T: Scalar>(inp: &VariableTensor, out : &VariableTensor, scale: u32, bit_length: u32, n_packed: u32, offset: T, var_dict: &mut Memory<T>) {
        let mut in_iter = inp.iter();
        let mut out_iter = out.iter();

        let base =  T::from_i32(1 << bit_length);
        let mut is_running = true;
        let mut step = scale;
        let mut cur_var = in_iter.next().unwrap();
        while is_running {
            let res = if let Some(r) = out_iter.next() {r} else {break};
            let mut pow = T::one();
            let mut sum_pow = T::zero();
            let mut sum_res = T::zero();
            for i in 0..n_packed {
                step -= 1;
                sum_pow += pow;
                pow *= base;
                if step == 0 || i == n_packed - 1{
                    sum_res += var_dict[cur_var as usize] * sum_pow;
                    sum_pow = T::zero();
                }
                if step == 0 {
                    step = scale;
                    match in_iter.next() {
                        Some(var) => cur_var = var,
                        None => {is_running = false; break}
                    }
                }
            }
            var_dict[res as usize] = sum_res * offset;
        }
    }
    pub fn run_packing_tensor<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (bit_length, n_packed) = (param[2] & 255, param[2] >> 8);
        let scale = param[3];
        let offset = if param.len() == 12 {
            T::slice_u32_to_scalar(&param[4..12])
        } else {
            T::one()
        };

        ConstraintSystem::run_packing(&mem[param[0]], &mem[param[1]], scale, bit_length, n_packed, offset, var_dict)
    }

    fn packing(&mut self, input: &VariableTensor, output: &VariableTensor, bit_length: u8, n_packed: u8, scale: u32, offset: BigScalar) {
        let mut in_iter = input.iter();
        let mut out_iter = output.iter();

        let base =  power_of_two::<BigScalar>(bit_length as u32);
        let mut is_running = true;
        let mut step = scale;
        let mut cur_var = in_iter.next().unwrap();
        while is_running {
            let res = if let Some(r) = out_iter.next() {r} else {break};
            let mut pow: BigScalar = BigScalar::one();
            let mut sum_pow = BigScalar::zero();
            for i in 0..n_packed {
                step -= 1;
                sum_pow += pow;
                pow *= base;
                if step == 0 || i == n_packed - 1{
                    self.a.push((self.n_cons, cur_var, sum_pow * offset));
                    sum_pow = BigScalar::zero();
                }
                if step == 0 {
                    step = scale;
                    match in_iter.next() {
                        Some(var) => cur_var = var,
                        None =>  {
                            is_running = false;
                            break;
                        }
                    }
                }
            }

            self.b.push((self.n_cons, self.mem.one_var, Scalar::one()));
            self.c.push((self.n_cons, res, Scalar::one()));
            self.n_cons += 1;
        }
    }

    pub fn packing_tensor(&mut self, input: TensorAddress, output: TensorAddress, bit_length: u8, n_packed: u8, scale: u32, offset: BigScalar, compute: bool) {
        self.packing(&self.mem[input].clone(), &self.mem[output].clone(), bit_length, n_packed, scale, offset);
        if compute {
            let mut params = vec![input, output, bit_length as u32 + ((n_packed as u32) << 8), scale];
            if offset != BigScalar::one() {
                params.extend_from_slice(&scalar_to_vec_u32(offset));
            }
            self.compute.push((params.into_boxed_slice(), Functions::Packing));
        }
    }

    pub fn run_packing_tensor_by_dim<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (bit_length, n_packed, dim_len) = (param[2] & 255, (param[2] >> 8) & 255, (param[2] >> 16));
        let dim_len = dim_len as usize;
        let scale = param[3];
        let mut dim = Vec::new();
        for &i in param[4..4 + dim_len].iter() {
            dim.push(i as i32);
        }
        let offset = if param.len() > 4 + dim_len {
            T::slice_u32_to_scalar(&param[4 + dim_len..12 + dim_len])
        } else {
            T::one()
        };

        for (input, output) in mem[param[0]].iter_slice(&dim).zip(mem[param[1]].iter_slice(&[-1])) {
            ConstraintSystem::run_packing(&input, &output, scale, bit_length, n_packed, offset, var_dict);
        }
    }

    pub fn packing_tensor_by_dim(&mut self, input: TensorAddress, dim: &[i32], output: TensorAddress, bit_length: u8, n_packed: u8, scale: u32, offset: BigScalar, compute: bool) {
        for (input, output) in self.mem[input].iter_slice(dim).zip(self.mem[output].iter_slice(&[-1])) {
            self.packing(&input, &output, bit_length, n_packed, scale, offset);
        }
        if compute {
            let packed_params = bit_length as u32 + ((n_packed as u32) << 8) + ((dim.len() as u32) << 16);
            let mut params = vec![input, output, packed_params, scale];
            for &d in dim {
                params.push(d as u32);
            }

            if offset != BigScalar::one() {
                params.extend_from_slice(&scalar_to_vec_u32(offset));
            }
            self.compute.push((params.into_boxed_slice(), Functions::PackingByDim));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::scalar::slice_to_scalar;

    #[test]
    fn packing_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[10]);
        let output = x.mem.alloc(&[2]);

        x.packing_tensor(input, output, 3, 5,1,BigScalar::one(), true);
        let mut mem = x.mem.new_memory::<i32>();
        x.load_memory(input, &mut mem, &[1,2,3,4,5,6,7,8,9,10]);

        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize..x.mem[output].end() as usize], [22737, 46142]);
        x.verify(&slice_to_scalar(&mem));
    }

    #[test]
    fn packing_test_2() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[4]);
        let output = x.mem.alloc(&[2]);

        x.packing_tensor(input, output, 3, 6,3,BigScalar::from(2u32), true);
        let mut mem = x.mem.new_memory::<i32>();
        x.load_memory(input, &mut mem, &[1,2,3,4]);

        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize..x.mem[output].end() as usize], [74825*2, 149723*2]);
        x.verify(&slice_to_scalar(&mem));
    }
}