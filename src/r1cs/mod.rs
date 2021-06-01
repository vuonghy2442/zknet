
use crate::tensor::VariableTensor;
use itertools::izip;
use crate::tensor::TensorIndex::Id;
use crate::tensor::TensorIndex::Range;
use crate::tensor::TensorIndex::RangeFull;
use std::ops::{Index};
use std::cmp::max;
use curve25519_dalek::scalar::Scalar;

pub fn to_scalar(val: i32) -> Scalar {
    if val < 0 {
        -Scalar::from((-val) as u32)
    } else {
        Scalar::from(val as u32)
    }
}

pub fn slice_to_scalar(data: &[i32]) -> Vec<Scalar> {
    let mut mem: Vec<Scalar> = Vec::new();
    for &val in data.iter() {
        mem.push(to_scalar(val));
    }
    mem
}

type ScalarAddress = u32;
pub type TensorAddress = u32;

union MemAddress {
    block_id: TensorAddress,
    memory_id: ScalarAddress
}

pub struct MemoryManager {
    mem_dict: Vec<VariableTensor>,
    n_var: u32,
    one_var: u32
}

impl MemoryManager {

    fn new() -> MemoryManager {
        MemoryManager {
            mem_dict: Vec::new(),
            n_var: 1,
            one_var: 0
        }
    }

    pub fn alloc(&mut self, shape: &[u32]) -> TensorAddress {
        let var = VariableTensor::new(self.n_var, shape);
        self.n_var += var.size();
        self.save(var)
    }

    fn alloc_single(&mut self) -> ScalarAddress {
        self.n_var += 1;
        return self.n_var - 1;
    }

    pub fn save(&mut self, tensor: VariableTensor) -> TensorAddress {
        self.mem_dict.push(tensor);
        (self.mem_dict.len() - 1) as TensorAddress
    }

    pub fn new_memory(&self) -> Vec<Scalar> {
        let mut var_dict: Vec<Scalar> = Vec::new();
        var_dict.resize(self.n_var as usize, Scalar::zero());
        var_dict[self.one_var as usize] = Scalar::one();
        var_dict
    }
}

impl Index<TensorAddress> for MemoryManager {
    type Output = VariableTensor;
    fn index(&self, idx: TensorAddress) -> &Self::Output {
        &self.mem_dict[idx as usize]
    }
}
type Memory = [Scalar];

pub struct ConstraintSystem {
    a: Vec<(u32, u32, Scalar)>,
    b: Vec<(u32, u32, Scalar)>,
    c: Vec<(u32, u32, Scalar)>,
    n_cons: u32,
    pub mem: MemoryManager,
    compute: Vec<(Box<[MemAddress]>, Functions)>
}

#[derive(Copy, Clone)]
enum Functions {
    Sum = 0,
    Mul = 1,
    Decompose = 2,
    Relu = 3,
    MaxPool = 4
}

impl ConstraintSystem {
    const FUNCTIONS: [fn(mem: &MemoryManager, &[MemAddress], &mut Memory); 5] = [
        Self::run_sum,
        Self::run_mul,
        Self::run_decompose,
        Self::run_relu,
        Self::run_max_pool
    ];

    pub fn new() -> ConstraintSystem {
        ConstraintSystem {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            n_cons: 0,
            mem: MemoryManager::new(),
            compute: Vec::new()
        }
    }

    // Bring input tensors to the end (please only put TensorAddress that is returned from alloc)
    pub fn reorder_for_spartan(&mut self, input: &[TensorAddress]) {
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        ranges.push((0,1)); //one var

        let mut total_input: u32 = 1;
        for &inp in input {
            let (l, r) = (self.mem[inp].begin(), self.mem[inp].end());
            total_input += r - l;
            ranges.push((l,r));
        }
        ranges.sort_unstable();
        let total_witness = self.mem.n_var - total_input;

        let mut offset: Vec<u32> = Vec::new();
        offset.push(0);
        for r in ranges.iter() {
            offset.push(offset.last().unwrap() + (r.1 - r.0));
        }

        let new_index = |pos: u32| {
            let mut l: usize = 0;
            let mut r: usize = ranges.len() - 1;
            let mut res: usize = 0;
            while l <= r {
                let mid = (l + r)/2;
                if ranges[mid].0 <= pos {
                    l = mid + 1;
                    res = mid;
                } else {
                    r = mid - 1;
                }
            }
            if pos < ranges[res].1 {
                // inside a moved block
                pos - ranges[res].0 + offset[res] + total_witness
            } else {
                pos  - offset[res+1]
            }
        };

        // Remap memdict
        self.mem.one_var = total_witness;
        for tensor in self.mem.mem_dict.iter_mut() {
            tensor.start = new_index(tensor.start);
        }

        // Remap a, b, c
        for a in self.a.iter_mut() { a.1 = new_index(a.1); }
        for b in self.b.iter_mut() { b.1 = new_index(b.1); }
        for c in self.c.iter_mut() { c.1 = new_index(c.1); }

        // Remap sum
        for (params, r) in self.compute.iter_mut() {
            if let Functions::Sum = r {
                for data in params[1..].iter_mut() {
                    data.memory_id = new_index(unsafe{data.memory_id});
                }
            }
        }
    }

    pub fn get_spartan_instance(&self) -> (libspartan::Instance, usize, usize, usize, usize) {
        fn parse_matrix(mat: &[(u32, u32, Scalar)]) -> Vec<(usize, usize, [u8; 32])> {
            let mut ans: Vec<(usize, usize, [u8; 32])> = Vec::with_capacity(mat.len());
            for row in mat {
                let u = row.0 as usize;
                let v = row.1 as usize;
                let w = row.2.to_bytes();
                ans.push((u, v, w))
            }
            return ans;
        }
        let num_cons = self.n_cons as usize;
        let num_vars = self.mem.one_var as usize;
        let num_inputs = (self.mem.n_var - self.mem.one_var - 1) as usize;
        let non_zeros=max(max(self.a.len(),self.b.len()),self.c.len());
        (libspartan::Instance::new(num_cons,
            num_vars,
            num_inputs,
            &parse_matrix(&self.a), &parse_matrix(&self.b), &parse_matrix(&self.c)
        ).unwrap(), num_cons, num_vars, num_inputs, non_zeros)
    }

    pub fn sort_cons(&mut self) {
        self.a.sort_unstable_by_key(|x| (x.0,x.1));
        self.b.sort_unstable_by_key(|x| (x.0,x.1));
        self.c.sort_unstable_by_key(|x| (x.0,x.1));
    }

    pub fn verify(&self, var_dict: &[Scalar]) -> bool {
        let (mut ai, mut bi, mut ci) = (0, 0, 0);
        for i in 0..self.n_cons {
            let (mut sa, mut sb, mut sc) = (Scalar::zero(), Scalar::zero(), Scalar::zero());
            while ai < self.a.len() && self.a[ai].0 == i {
                sa += var_dict[self.a[ai].1 as usize] * self.a[ai].2;
                ai += 1;
            }
            while bi < self.b.len() && self.b[bi].0 == i {
                sb += var_dict[self.b[bi].1 as usize] * self.b[bi].2;
                bi += 1;
            }
            while ci < self.c.len() && self.c[ci].0 == i {
                sc += var_dict[self.c[ci].1 as usize] * self.c[ci].2;
                ci += 1;
            }

            if sa * sb != sc {
                println!("{} {} {}", self.a[ai-1].1, self.b[bi-1].1, self.c[ci-1].1);
                println!("{} {} {}", sa.as_bytes()[0], sb.as_bytes()[0], sc.as_bytes()[0]);
                return false;
            }
        }
        return true;
    }

    pub fn load_memory(&self, tensor: TensorAddress, var_dict: &mut Memory, data: &[Scalar]) {
        assert_eq!(self.mem[tensor].size(), data.len() as u32);
        for (pos, &data) in izip!(self.mem[tensor].iter(), data) {
            var_dict[pos as usize] = data;
        }
    }

    pub fn compute(&self, var_dict: &mut Memory) {
        for (params, func) in self.compute.iter() {
            Self::FUNCTIONS[*func as usize](&self.mem, &params, var_dict);
        }
    }

    pub fn cons_size(&self) -> u32 {
        return self.n_cons;
    }

    fn run_mul(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
        let (a, b, res) = unsafe{(param[0].block_id, param[1].block_id, param[2].block_id)};
        for (x, y, z) in izip!(mem[a].iter(), mem[b].iter(), mem[res].iter()) {
            var_dict[z as usize] = var_dict[x as usize] * var_dict[y as usize];
        }
    }

    pub fn mul(&mut self, a: TensorAddress, b: TensorAddress, res: TensorAddress) {
        for (x, y, z) in izip!(self.mem[a].iter(), self.mem[b].iter(), self.mem[res].iter()) {
            self.a.push((self.n_cons, x, Scalar::one()));
            self.b.push((self.n_cons, y, Scalar::one()));
            self.c.push((self.n_cons, z, Scalar::one()));
            self.n_cons += 1;
        }

        self.compute.push((Box::new([MemAddress{block_id: a}, MemAddress{block_id: b}, MemAddress{block_id: res}]), Functions::Mul));
    }

    fn run_sum(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
        let mut res: Scalar = if param.len() == 3 {var_dict[unsafe {param[2].memory_id} as usize]} else {Scalar::zero()};
        for x in mem[unsafe {param[0].block_id}].iter() {
            res += var_dict[x as usize];
        }
        var_dict[unsafe {param[1].memory_id} as usize] = res;
    }

    pub fn sum(&mut self, inp: TensorAddress, out: ScalarAddress, init: Option<u32>) {
        if let Some(x) = init {
            self.a.push((self.n_cons, x, Scalar::one()));
        }

        for x in self.mem[inp].iter() {
            self.a.push((self.n_cons, x, Scalar::one()));
        }
        self.b.push((self.n_cons, self.mem.one_var, Scalar::one()));
        self.c.push((self.n_cons, out, Scalar::one()));
        self.n_cons += 1;

        let mut params = vec![MemAddress{block_id: inp}, MemAddress{memory_id: out}];
        if let Some(x) = init {
            params.push(MemAddress{memory_id: x});
        }
        self.compute.push((params.into_boxed_slice(), Functions::Sum));
    }

    pub fn conv2d(&mut self, input: TensorAddress, output: TensorAddress, weight: TensorAddress, bias: Option<(TensorAddress, u32)>) {
        let fout = self.mem[weight].dim[0];
        let fin = self.mem[weight].dim[1];
        let kx = self.mem[weight].dim[2];
        let ky = self.mem[weight].dim[3];
        let (out_row, out_col) = (self.mem[input].dim[1] - kx + 1, self.mem[input].dim[2] - ky + 1);
        let mut cur_weight: Vec<TensorAddress> = Vec::with_capacity(fout as usize);
        for layer in 0..fout {
            cur_weight.push(self.mem.save(self.mem[weight].at(&[Id(layer)])));
        }

        let mut cur_input: Vec<Vec<TensorAddress>> = Vec::with_capacity(out_row as usize);
        for i in 0..out_row {
            let mut tmp: Vec<TensorAddress> = Vec::with_capacity(out_col as usize);
            for j in 0..out_col {
                tmp.push(self.mem.save(self.mem[input].at(&[RangeFull(), Range(i..i+kx), Range(j..j+ky)])));
            }
            cur_input.push(tmp);
        }

        for layer in 0..fout {
            for i in 0..out_row{
                for j in 0..out_col{
                    let tmp = self.mem.alloc(&[fin, kx, ky]);
                    self.mul(cur_input[i as usize][j as usize], cur_weight[layer as usize], tmp);
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

    // Becareful when use this
    fn positive(x: &Scalar) -> bool {
        return (x.as_bytes()[31] >> 4) == 0;
    }

    fn run_decompose(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
        let (input, output, sign, abs) = unsafe{(param[0].block_id, param[1].block_id, param[2].block_id, param[3].block_id)};
        let mut iter = mem[input].iter();
        loop {
            let x = iter.next();
            if let None = x {
                break
            }
            let x = var_dict[x.unwrap() as usize];
            let idx = &iter.idx;
            let bits = mem[output].at_(idx);
            let sign = mem[sign].at_idx(idx);
            let abs = mem[abs].at_idx(idx);

            let positive = Self::positive(&x);

            var_dict[sign as usize] = if positive {Scalar::one()} else {-Scalar::one()};
            var_dict[abs as usize] = if positive {x} else {-x};

            let data = var_dict[abs as usize].reduce().to_bytes();
            for (i, bit) in bits.iter().enumerate() {
                var_dict[bit as usize] = Scalar::from((data[i/8] >> (i%8)) & 1);
            }
        }
    }

    // input should have shape with sign, abs, and output should have one more dimension with length bit size
    pub fn bit_decomposition(&mut self, input: TensorAddress, output: TensorAddress, sign: TensorAddress, abs: TensorAddress) {
        let mut iter = self.mem[input].iter();
        loop {
            let x = iter.next();
            if let None = x {
                break
            }
            let x = x.unwrap();
            let idx = &iter.idx;

            let bits = self.mem[output].at_(idx);
            let abs = self.mem[abs].at_idx(idx);
            let sign = self.mem[sign].at_idx(idx);
            // abs * sign == input
            self.a.push((self.n_cons, abs, Scalar::one()));
            self.b.push((self.n_cons, sign, Scalar::one()));
            self.c.push((self.n_cons, x, Scalar::one()));
            self.n_cons += 1;

            // sign only = +-1 <=> sign^2 == 1
            self.a.push((self.n_cons, sign, Scalar::one()));
            self.b.push((self.n_cons, sign, Scalar::one()));
            self.c.push((self.n_cons, self.mem.one_var, Scalar::one()));
            self.n_cons += 1;

            let sum_cons = self.n_cons;
            self.n_cons += 1;

            let mut pow: u32 = 1;
            for bit in bits.iter() {
                self.a.push((sum_cons, bit, Scalar::from(pow)));
                pow *= 2;

                // bit only 0 or 1 <=> bit^2 = bit
                self.a.push((self.n_cons, bit, Scalar::one()));
                self.b.push((self.n_cons, bit, Scalar::one()));
                self.c.push((self.n_cons, bit, Scalar::one()));
                self.n_cons += 1;
            }
        }


        let params = Box::new([MemAddress{block_id: input}, MemAddress{block_id: output}, MemAddress{block_id: sign}, MemAddress{block_id: abs}]);
        self.compute.push((params, Functions::Decompose));
    }

    pub fn sign(&mut self, input: TensorAddress, output: TensorAddress, max_bits: u8) {
        let mut dim = self.mem[input].dim.to_vec();
        let abs = self.mem.alloc(&dim);
        dim.push(max_bits as u32);
        let bits = self.mem.alloc(&dim);
        self.bit_decomposition(input, bits, output, abs);
    }

    fn run_relu(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
        for (input, output) in izip!(mem[unsafe {param[0].block_id}].iter(), mem[unsafe {param[1].block_id}].iter()) {
            let x = var_dict[input as usize];
            var_dict[output as usize] = if Self::positive(&x) {x} else {Scalar::zero()}; // hic hic here
        }
    }

    pub fn relu(&mut self, input: TensorAddress, output: TensorAddress, max_bits: u8) {
        let mut dim = self.mem[input].dim.to_vec();
        let abs = self.mem.alloc(&dim);
        let sign = self.mem.alloc(&dim);
        dim.push(max_bits as u32);
        let bits = self.mem.alloc(&dim);
        self.bit_decomposition(input, bits, sign, abs);

        for (abs, input, output) in izip!(self.mem[abs].iter(), self.mem[input].iter(), self.mem[output].iter()) {
            self.a.push((self.n_cons, input, Scalar::one()));
            self.a.push((self.n_cons, abs, Scalar::one()));
            self.a.push((self.n_cons, output, Scalar::from(2u32)));
            self.n_cons += 1;
        }

        let params = Box::new([MemAddress{block_id: input}, MemAddress{block_id: output}]);
        self.compute.push((params, Functions::Relu));
    }

    fn shift_right(x: &Scalar, k: u8) -> Scalar {
        let mut res = x.reduce().to_bytes();
        for i in 0..res.len()-1 {
            res[i] = (res[i] >> k) | (res[i+1] << (8 - k));
        }
        res[res.len()-1] >>= k;
        Scalar::from_bits(res)
    }

    fn run_max_pool(mem: &MemoryManager, param: &[MemAddress], var_dict: &mut Memory) {
        let (input, output, temp) = unsafe{(param[0].block_id, param[1].block_id, param[2].block_id)};
        for layer in 0..mem[input].dim[0] {
            let input = mem[input].at_(&[layer]);
            let output = mem[output].at_(&[layer]);
            for i in 0..input.dim[0]/2 {
                for j in 0..input.dim[1]/2 {
                    let t = [mem[temp].at_idx(&[layer, i, j, 0]), mem[temp].at_idx(&[layer, i, j, 1])];
                    let mut val = [Scalar::zero(),Scalar::zero()];
                    for k in 0..2 {
                        val[k] = if var_dict[input.at_idx(&[2*i + k as u32, 2*j]) as usize] == Scalar::one() ||
                                    var_dict[input.at_idx(&[2*i + k as u32, 2*j + 1]) as usize] == Scalar::one() {
                            Scalar::zero()
                        } else {
                            Scalar::from(2u8)
                        };
                        var_dict[t[k] as usize] = val[k];
                    }
                    var_dict[output.at_idx(&[i,j]) as usize] = if val[0] == Scalar::zero() || val[1] == Scalar::zero() {
                        Scalar::one()
                    } else {
                        -Scalar::one()
                    };
                }
            }
        }
    }

    // input tensor 3d
    pub fn binary_max_pool(&mut self, input: TensorAddress, output: TensorAddress) {
        let mut dim = self.mem[input].dim.to_vec();
        dim.push(2);
        let temp = self.mem.alloc(&dim);
        for layer in 0..self.mem[input].dim[0] {
            let input = self.mem[input].at_(&[layer]);
            let output = self.mem[output].at_(&[layer]);
            for i in 0..input.dim[0]/2 {
                for j in 0..input.dim[1]/2 {
                    let t = [self.mem[temp].at_idx(&[layer, i, j, 0]), self.mem[temp].at_idx(&[layer, i, j, 1])];
                    for k in 0..2 {
                        self.a.push((self.n_cons, input.at_idx(&[2*i + k, 2*j]), Scalar::one()));
                        self.a.push((self.n_cons, self.mem.one_var, -Scalar::one()));
                        self.b.push((self.n_cons, input.at_idx(&[2*i + k, 2*j + 1]), Scalar::one()));
                        self.b.push((self.n_cons, self.mem.one_var, -Scalar::one()));
                        self.c.push((self.n_cons, t[k as usize], Scalar::from(2u32)));
                        self.n_cons += 1;
                    }
                    self.a.push((self.n_cons, t[0], Scalar::one()));
                    self.b.push((self.n_cons, t[1], Scalar::one()));
                    self.c.push((self.n_cons,  output.at_idx(&[i,j]), -Scalar::from(2u32)));
                    self.c.push((self.n_cons,  self.mem.one_var, Scalar::from(2u32)));
                    self.n_cons += 1;
                }
            }
        }

        let params = Box::new([MemAddress{block_id: input}, MemAddress{block_id: output}, MemAddress{block_id: temp}]);
        self.compute.push((params, Functions::MaxPool));
    }

    pub fn fully_connected(&mut self, input: TensorAddress, output: TensorAddress, weight: TensorAddress, bias: Option<TensorAddress>) {
        let temp = self.mem.alloc(&self.mem[weight].dim.clone());
        for i in 0..self.mem[weight].dim[0] {
            let temp = self.mem.save(self.mem[temp].at_(&[i]));
            let weight = self.mem.save(self.mem[weight].at_(&[i]));
            self.mul(weight, input, temp);
            match bias {
                Some(b) => self.sum(temp, self.mem[output].at_idx(&[i]), Some(self.mem[b].at_idx(&[i]))),
                None => self.sum(temp, self.mem[output].at_idx(&[i]), None)
            }
        }
    }

    pub fn compress_binary_weight(&mut self, input: TensorAddress, output: TensorAddress, bit_length: u8) {
        let mut in_iter = self.mem[input].iter();
        let mut out_iter = self.mem[output].iter();

        loop {
            let mut pow = 1;
            for i in 0..bit_length {

            }
        }
    }
}

struct PoseidonHash {

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_cons_test() {
        let mut x = ConstraintSystem::new();
        let a = x.mem.alloc(&[3]);
        let b = x.mem.alloc(&[3]);
        let c = x.mem.alloc(&[3]);
        x.mul(a, b, c);

        let mut mem: Vec<Scalar> = slice_to_scalar(&[1,5,2,3,1,6,3,0,0,0]);
        x.compute(&mut mem);
        assert_eq!(mem[7..10], slice_to_scalar(&[5, 12, 9]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn sum_cons_test() {
        let mut x = ConstraintSystem::new();
        let a = x.mem.alloc(&[3]);
        let bias = x.mem.alloc_single();
        let res = x.mem.alloc_single();

        x.sum(a, res, Some(bias));

        let mut mem: Vec<Scalar> = slice_to_scalar(&[1,5,2,3,-2,0]);
        x.compute(&mut mem);
        assert_eq!(mem[5], Scalar::from(8u32));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn conv2d_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,5,5]);
        let weight = x.mem.alloc(&[2,2,3,3]);
        let output = x.mem.alloc(&[2,3,3]);
        let bias = x.mem.alloc(&[2,3,3]);

        x.conv2d(input, output, weight, Some((bias, 1)));

        let mut mem: Vec<Scalar> = slice_to_scalar(&[1,0,1,-1,0,0,0,-2,4,-1,-4,0,3,-4,0,0,0,1,-1,1,-4,2,3,-1,0,-4,2,2,-3,-1,-1,1,2,-1,1,4,4,2,3,-3,0,3,-2,3,0,2,3,3,-2,2,4,3,3,-4,-4,-1,3,1,4,-2,-2,0,-2,4,-3,0,0,0,-2,0,0,0,0,3,4,-3,-4,-1,-1,-4,3,1,-2,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,0,-3,0,1,-4,-1,2,0,0,-4,2,1,3,2,-3,4,-3]);
        mem.resize(x.mem.n_var as usize, Scalar::zero());

        x.compute(&mut mem);
        assert_eq!(mem[87..87+18], slice_to_scalar(&[32,3,-36,-27,-9,59,44,-21,-16,-23,25,-4,-24,-8,21,-15,-33,-1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn sign_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,2]);
        let output = x.mem.alloc(&[2,2]);
        x.sign(input, output, 3);
        let mut mem = x.mem.new_memory();
        mem[1..5].copy_from_slice(&slice_to_scalar(&[5,-2,3,-4]));
        x.compute(&mut mem);
        assert_eq!(mem[5..9],slice_to_scalar(&[1,-1,1,-1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn relu_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,2]);
        let output = x.mem.alloc(&[2,2]);
        x.relu(input, output, 3);
        let mut mem = x.mem.new_memory();
        mem[1..5].copy_from_slice( &slice_to_scalar(&[5,-2,3,-4]));
        x.compute(&mut mem);
        assert_eq!(mem[5..9],slice_to_scalar(&[5,0,3,0]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn max_pool_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,2,4]);
        let output = x.mem.alloc(&[2,1,2]);
        x.binary_max_pool(input, output);
        let mut mem = x.mem.new_memory();
        mem[1..17].copy_from_slice(&slice_to_scalar(&[1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1]));
        x.compute(&mut mem);
        assert_eq!(mem[17..21],slice_to_scalar(&[1,-1,1,1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn fully_connected_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[5]);
        let output = x.mem.alloc(&[2]);
        let weight = x.mem.alloc(&[2,5]);
        let bias = x.mem.alloc(&[2]);
        x.fully_connected(input, output, weight, Some(bias));

        let mut mem = x.mem.new_memory();
        mem[1..6].copy_from_slice(&slice_to_scalar(&[2,-2,4,3,1]));
        mem[8..18].copy_from_slice(&slice_to_scalar(&[-2,3,-2,5,3,-1,5,0,3,2]));
        x.compute(&mut mem);
        assert_eq!(mem[6..8], slice_to_scalar(&[0, -1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn move_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[5]);
        let output = x.mem.alloc(&[2]);
        let weight = x.mem.alloc(&[2,5]);
        let bias = x.mem.alloc(&[2]);
        x.fully_connected(input, output, weight, Some(bias));
        x.reorder_for_spartan(&[input, output]);

        let mut mem = x.mem.new_memory();
        println!("{} {} {}", x.mem[input].begin(), x.mem[input].end(), x.mem.one_var);
        println!("{} {} {}", x.mem[output].begin(), x.mem[output].end(), x.mem.one_var);

        x.load_memory(input, &mut mem,  &slice_to_scalar(&[2,-2,4,3,1]));
        x.load_memory(weight, &mut mem, &slice_to_scalar(&[-2,3,-2,5,3,-1,5,0,3,2]));
        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize .. x.mem[output].end() as usize], slice_to_scalar(&[0, -1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn scalar_test() {
        let scalar = -Scalar::one();
        for i in scalar.as_bytes() {
            print!("{} ", i);
        }
    }
}