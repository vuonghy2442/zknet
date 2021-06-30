use std::{convert::TryInto, usize};

use super::{ConstraintSystem, Scalar, MemoryManager, Memory, TensorAddress, BigScalar, Functions};

impl ConstraintSystem {
    pub fn run_sum_pool<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        if let [input, output, krow, kcol] = *param {
            let dim: [u32;3] = (*mem[input].dim).try_into().unwrap();
            let [layer, row, col] = dim;
            for k in 0..layer {
                for i in 0..row/krow {
                    for j in 0..col/kcol {
                        let mut s = T::zero();
                        for ii in 0..krow {
                            for jj in 0..kcol {
                                s += var_dict[mem[input].at_idx(&[k, i * krow + ii, j * kcol + jj]) as usize];
                            }
                        }
                        var_dict[mem[output].at_idx(&[k, i, j]) as usize] = s;
                    }
                }
            }
        } else {
            panic!("params don't match");
        }
    }
    pub fn sum_pool(&mut self, input: TensorAddress, output: TensorAddress, kernel: [u32;2]) {
        let dim: [u32;3] = (*self.mem[input].dim).try_into().unwrap();
        let [layer, row, col] = dim;
        for k in 0..layer {
            for i in 0..row/kernel[0] {
                for j in 0..col/kernel[1] {
                    for ii in 0..kernel[0] {
                        for jj in 0..kernel[1] {
                            self.a.push((self.n_cons,self.mem[input].at_idx(&[k, i * kernel[0] + ii, j * kernel[1] + jj]),BigScalar::one()));
                        }
                    }
                    self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
                    self.c.push((self.n_cons,self.mem[output].at_idx(&[k, i, j]), BigScalar::one()));
                    self.n_cons += 1;
                }
            }
        }
        self.compute.push((Box::new([input,output,kernel[0],kernel[1]]), Functions::SumPool));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn average_pool_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,4,4]);
        let output = x.mem.alloc(&[2,2,2]);
        x.sum_pool(input, output, [2,2]);
    }
}