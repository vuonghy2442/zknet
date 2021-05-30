
use crate::tensor::VariableTensor;
use itertools::izip;
use crate::tensor::TensorIndex::Id;
use crate::tensor::TensorIndex::Range;
use crate::tensor::TensorIndex::RangeFull;
use std::ops::{Index, IndexMut};

type Scalar = i32;
type CellAddress = u32;
type BlockAddress = u32;

union MemAddress {
    block_id: BlockAddress,
    memory_id: CellAddress
}

struct MemoryManager {
    mem_dict: Vec<VariableTensor>,
    n_var: u32,
}

impl MemoryManager {
    const ONE_VAR: u32 = 0;

    fn new() -> MemoryManager {
        MemoryManager {
            mem_dict: Vec::new(),
            n_var: 1
        }
    }

    fn alloc(&mut self, shape: &[u32]) -> BlockAddress {
        let var = VariableTensor::new(self.n_var, shape);
        self.n_var += var.size();
        self.mem_dict.push(var);
        (self.mem_dict.len() - 1) as BlockAddress
    }

    fn alloc_single(&mut self) -> CellAddress {
        self.n_var += 1;
        return self.n_var - 1;
    }

    fn save(&mut self, tensor: VariableTensor) -> BlockAddress {
        self.mem_dict.push(tensor);
        (self.mem_dict.len() - 1) as BlockAddress
    }

    fn new_memory(&self) -> Vec<Scalar> {
        let mut var_dict: Vec<Scalar> = Vec::new();
        var_dict.resize(self.n_var as usize, 0);
        var_dict[Self::ONE_VAR as usize] = 1;
        var_dict
    }
}

impl Index<BlockAddress> for MemoryManager {
    type Output = VariableTensor;
    fn index(&self, idx: BlockAddress) -> &Self::Output {
        &self.mem_dict[idx as usize]
    }
}
type Memory = [Scalar];

struct ComputationCircuit {
    a: Vec<(u32, u32, i32)>,
    b: Vec<(u32, u32, i32)>,
    c: Vec<(u32, u32, i32)>,
    n_cons: u32,
    mem: MemoryManager,
    compute: Vec<(Vec<MemAddress>, fn(mem: &MemoryManager, &[MemAddress], &mut Memory))>
}

impl ComputationCircuit {
    const ONE_VAR: u32 = 0;

    fn new() -> ComputationCircuit {
        ComputationCircuit {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            n_cons: 0,
            mem: MemoryManager::new(),
            compute: Vec::new()
        }
    }

    fn compute(&self, var_dict: &mut Memory) {
        for (params, func) in &self.compute {
            func(&self.mem, &params, var_dict);
        }
    }

    fn mul(&mut self, a: BlockAddress, b: BlockAddress, res: BlockAddress) {
        for (x, y, z) in izip!(self.mem[a].iter(), self.mem[b].iter(), self.mem[res].iter()) {
            self.a.push((self.n_cons, x, 1));
            self.b.push((self.n_cons, y, 1));
            self.c.push((self.n_cons, z, 1));
            self.n_cons += 1;
        }
        fn run(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
            let (a, b, res) = unsafe{(param[0].block_id, param[1].block_id, param[2].block_id)};
            for (x, y, z) in izip!(mem[a].iter(), mem[b].iter(), mem[res].iter()) {
                var_dict[z as usize] = var_dict[x as usize] * var_dict[y as usize];
            }
        }
        self.compute.push((vec![MemAddress{block_id: a}, MemAddress{block_id: b}, MemAddress{block_id: res}], run));
    }

    fn sum(&mut self, inp: BlockAddress, out: CellAddress, init: Option<u32>) {
        if let Some(x) = init {
            self.a.push((self.n_cons, x, 1));
        }

        for x in self.mem[inp].iter() {
            self.a.push((self.n_cons, x, 1));
        }
        self.b.push((self.n_cons, MemoryManager::ONE_VAR, 1));
        self.c.push((self.n_cons, out, 1));
        self.n_cons += 1;

        fn run(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
            let mut res: Scalar = if param.len() == 3 {var_dict[unsafe {param[2].memory_id} as usize]} else {0};
            for x in mem[unsafe {param[0].block_id}].iter() {
                res += var_dict[x as usize];
            }
            var_dict[unsafe {param[1].memory_id} as usize] = res;
        }
        let mut params = vec![MemAddress{block_id: inp}, MemAddress{memory_id: out}];
        if let Some(x) = init {
            params.push(MemAddress{memory_id: x});
        }
        self.compute.push((params, run));
    }

    fn conv2d(&mut self, input: BlockAddress, output: BlockAddress, weight: BlockAddress, bias: Option<(BlockAddress, u32)>) {
        let fout = self.mem[weight].dim[0];
        let fin = self.mem[weight].dim[1];
        let kx = self.mem[weight].dim[2];
        let ky = self.mem[weight].dim[3];
        for layer in 0..fout {
            let cur_weight = self.mem.save(self.mem[weight].at(&[Id(layer)]));
            for i in 0..self.mem[input].dim[1] - kx + 1 {
                for j in 0..self.mem[input].dim[2] - ky + 1 {
                    let tmp = self.mem.alloc(&[fin, kx, ky]);
                    let cur_input = self.mem.save(self.mem[input].at(&[RangeFull(), Range(i..i+kx), Range(j..j+ky)]));
                    self.mul(cur_input, cur_weight, tmp);
                    let cur_bias = if let Some((b, scale)) = bias {
                        Some(self.mem[b].at_idx(&[layer, i/scale, j/scale]))
                    } else {
                        None
                    };
                    self.sum(tmp, self.mem[output].at_idx(&[layer, i, j]), cur_bias);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_circuit_test() {
        let mut x = ComputationCircuit::new();
        let a = x.mem.alloc(&[3]);
        let b = x.mem.alloc(&[3]);
        let c = x.mem.alloc(&[3]);
        x.mul(a, b, c);

        let mut mem: Vec<i32> = vec![1,5,2,3,1,6,3,0,0,0];
        x.compute(&mut mem);
        assert_eq!(mem[7..10], [5, 12, 9]);
    }

    #[test]
    fn sum_circuit_test() {
        let mut x = ComputationCircuit::new();
        let a = x.mem.alloc(&[3]);
        let bias = x.mem.alloc_single();
        let res = x.mem.alloc_single();

        x.sum(a, res, Some(bias));

        let mut mem: Vec<i32> = vec![1,5,2,3,-2,0];
        x.compute(&mut mem);
        assert_eq!(mem[5], 8);
    }

    #[test]
    fn conv2d_test() {
        let mut x = ComputationCircuit::new();
        let input = x.mem.alloc(&[2,5,5]);
        let weight = x.mem.alloc(&[2,2,3,3]);
        let output = x.mem.alloc(&[2,3,3]);
        let bias = x.mem.alloc(&[2,3,3]);

        x.conv2d(input, output, weight, Some((bias, 1)));

        let mut mem: Vec<i32> = vec![1,0,1,-1,0,0,0,-2,4,-1,-4,0,3,-4,0,0,0,1,-1,1,-4,2,3,-1,0,-4,2,2,-3,-1,-1,1,2,-1,1,4,4,2,3,-3,0,3,-2,3,0,2,3,3,-2,2,4,3,3,-4,-4,-1,3,1,4,-2,-2,0,-2,4,-3,0,0,0,-2,0,0,0,0,3,4,-3,-4,-1,-1,-4,3,1,-2,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,0,-3,0,1,-4,-1,2,0,0,-4,2,1,3,2,-3,4,-3];
        mem.resize(x.mem.n_var as usize, 0);

        x.compute(&mut mem);
        assert_eq!(mem[87..87+18], [32,3,-36,-27,-9,59,44,-21,-16,-23,25,-4,-24,-8,21,-15,-33,-1]);
    }
}