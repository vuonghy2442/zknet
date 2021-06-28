use super::{ConstraintSystem, Scalar, MemoryManager, Memory, TensorAddress, scalar, BigScalar, min, Functions};
use crate::scalar::power_of_two;

impl ConstraintSystem {

    pub fn run_fully_connected_compact<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        if let [input, output, weight, packed_weight, sum, max_bits] = param[..6] {

            let n_packed = scalar::SCALAR_SIZE/max_bits as u32;
            let input_size = mem[input].size();
            let output_size = mem[output].size();
            let output_packed = (output_size - 1)/n_packed + 1;

            //packing and multiply
            let mut input_iter = mem[input].iter();
            for i in 0..input_size {
                let cur_input = input_iter.next().unwrap();
                for j in 0..output_packed {
                    let packed = mem[packed_weight].at_idx(&[j, i]);
                    let mut s = T::zero();
                    for k in n_packed*j..min(output_size, n_packed*(j+1)) {
                        s += power_of_two::<T>((k - n_packed * j) * max_bits) * var_dict[mem[weight].at_idx(&[k, i]) as usize];
                    }
                    s *= var_dict[cur_input as usize];
                    var_dict[packed as usize] = s;
                }
            }

            // calculate sum
            let mut bias_iter = if param.len() == 7 {
                Some(mem[param[6]].iter())
            } else {
                None
            };

            for i in 0..output_packed {
                //prepare bias
                let mut s: T = T::zero();
                if let Some(iter) = bias_iter.as_mut() {
                    for k in 0..n_packed {
                        match iter.next() {
                            Some(x) => {
                                s += var_dict[x as usize] * power_of_two(k * max_bits);
                            },
                            None => break
                        }
                    }
                }
                for j in 0..input_size {
                    s += var_dict[mem[packed_weight].at_idx(&[i,j]) as usize];
                }
                var_dict[mem[sum].at_idx(&[i]) as usize] = s;
            }

            let offset = power_of_two::<T>(max_bits - 1);
            let mut big_offset = T::zero();
            for _ in 0..n_packed{
                big_offset = (big_offset * T::from_i32(2) + T::one()) * offset;
            }

            //matching result
            let mut output_iter = mem[output].iter();
            for i in 0..output_packed {
                let val = (var_dict[mem[sum].at_idx(&[i]) as usize] + big_offset).to_bytes();
                let mut ext = Vec::new();
                ext.resize(n_packed as usize, T::zero());
                for k in 0..n_packed * max_bits {
                    ext[(k / max_bits) as usize] += T::from_i32(((val[(k/8) as usize] >> (k % 8)) & 1) as i32) * power_of_two(k % max_bits);
                }
                for k in 0..n_packed {
                    let idx = i * n_packed + k;
                    if idx >= output_size {
                        break
                    }
                    let output_pos = output_iter.next().unwrap();
                    var_dict[output_pos as usize] = ext[k as usize] - offset;
                }
            }
        } else {
            panic!("params don't match");
        }
    }

    // Return out sign tensor
    pub fn fully_connected_compact(&mut self, input: TensorAddress, output: TensorAddress, weight: TensorAddress, bias: Option<TensorAddress>, max_bits: u8, relu: bool) -> TensorAddress {
        let n_packed = scalar::SCALAR_SIZE/max_bits as u32;
        let input_size = self.mem[input].size();
        let output_size = self.mem[output].size();
        let output_packed = (output_size - 1)/n_packed + 1;

        //packing and multiply
        let packed_weight = self.mem.alloc(&[output_packed, input_size]);
        let mut input_iter = self.mem[input].iter();
        for i in 0..input_size {
            let cur_input = input_iter.next().unwrap();
            for j in 0..output_packed {
                let packed = self.mem[packed_weight].at_idx(&[j, i]);
                for k in n_packed*j..min(output_size, n_packed*(j+1)) {
                    self.a.push((self.n_cons, self.mem[weight].at_idx(&[k, i]), power_of_two((k - n_packed * j) * max_bits as u32)));
                }
                self.b.push((self.n_cons, cur_input, BigScalar::one()));
                self.c.push((self.n_cons, packed, BigScalar::one()));
                self.n_cons += 1;
            }
        }

        // calculate sum
        let mut bias_iter = if let Some(b) =  bias {
            Some(self.mem[b].iter())
        } else {
            None
        };

        let sum = self.mem.alloc(&[output_packed]);
        for i in 0..output_packed {
            //prepare bias
            if let Some(iter) = bias_iter.as_mut() {
                for k in 0..n_packed {
                    match iter.next() {
                        Some(x) => {
                            self.a.push((self.n_cons, x, power_of_two(k * max_bits as u32)));
                        },
                        None => break
                    }
                }
            }
            for j in 0..input_size {
                self.a.push((self.n_cons, self.mem[packed_weight].at_idx(&[i,j]), BigScalar::one()));
                            }
            self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
            self.c.push((self.n_cons, self.mem[sum].at_idx(&[i]), BigScalar::one()));
            self.n_cons += 1;
        }

        // bit decomposition
        let sign = self.mem.alloc(&[output_size]);
        self.packing_tensor(output, sum, max_bits, n_packed as u8, 1, BigScalar::one(), false);

        let mut params = vec![input, output, weight, packed_weight, sum, max_bits as u32];
        if let Some(b) = bias {
            params.push(b);
        }
        self.compute.push((params.into_boxed_slice(), Functions::FCCompact));
        if relu {
            self.relu(output, sign, max_bits - 1);
        } else {
            self.sign(output, sign, max_bits - 1);
        }
        sign
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use scalar::slice_to_scalar;

    #[test]
    fn fully_connected_compact_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[5]);
        let output = x.mem.alloc(&[2]);
        let weight = x.mem.alloc(&[2,5]);
        let bias = x.mem.alloc(&[2]);
        x.fully_connected_compact(input, output, weight, Some(bias),4, false);

        let mut mem = x.mem.new_memory();
        mem[1..6].copy_from_slice(&slice_to_scalar(&[2,-2,4,3,1]));
        mem[8..18].copy_from_slice(&slice_to_scalar(&[-2,3,-2,5,3,-1,5,0,3,2]));
        x.load_memory(bias, &mut mem, &slice_to_scalar(&[5,-1]));
        x.compute(&mut mem);
        assert_eq!(mem[6..8], slice_to_scalar(&[5, -2]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
}