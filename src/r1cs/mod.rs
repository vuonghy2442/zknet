
use crate::tensor::VariableTensor;
use itertools::izip;
use crate::tensor::TensorIndex::Id;
use crate::tensor::TensorIndex::Range;
use crate::tensor::TensorIndex::RangeFull;
use std::fmt::Debug;
use std::ops::{Index};
use std::cmp::{min, max};
use curve25519_dalek::scalar::Scalar as BigScalar;

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

type ScalarAddress = u32;
pub type TensorAddress = u32;

type Memory<T> = [T];

pub trait Functional: Sized {
    const FUNCTIONS: [fn(mem: &MemoryManager, &[u32], &mut [Self]); 7];
}

pub trait Scalar: Debug + std::ops::Mul<Self, Output=Self> + std::ops::MulAssign<Self> + std::ops::Add<Self, Output=Self> + std::ops::AddAssign<Self> + std::ops::Sub<Self, Output=Self> + std::ops::Neg<Output=Self> + std::cmp::Eq + Functional + Clone + Copy {
    fn one() -> Self;
    fn zero() -> Self;
    fn from_i32(x: i32) -> Self;
    fn is_nonneg(&self) -> bool;
    fn to_bytes(&self) -> Vec<u8>;
    fn to_big_scalar(x: &[Self]) -> Vec<BigScalar>;
    fn slice_u32_to_scalar(x: &[u32]) -> Self;
}

fn pack_four_bytes(x : &[u8]) -> u32 {
    return (x[0] as u32) | (x[1] as u32) << 8 | (x[2] as u32) << 16 | (x[3] as u32) << 24;
}

fn scalar_to_vec_u32(x: BigScalar) -> [u32;8]{
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

fn power_of_two(x: u32) -> BigScalar {
    let mut t = [0u8;32];
    t[(x/8) as usize] = 1u8 << (x%8);
    BigScalar::from_bits(t)
}

const SCALAR_SIZE: u32 = 252;

impl<T:Scalar> Functional for T {
    const FUNCTIONS: [fn(mem: &MemoryManager, &[u32], &mut [T]); 7] = [
        ConstraintSystem::run_sum::<T>,
        ConstraintSystem::run_mul::<T>,
        ConstraintSystem::run_decompose::<T>,
        ConstraintSystem::run_relu::<T>,
        ConstraintSystem::run_max_pool::<T>,
        ConstraintSystem::run_packing_tensor::<T>,
        ConstraintSystem::run_conv2d_compact::<T>
    ];
}

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

    pub fn new_memory<T : Scalar>(&self) -> Vec<T> {
        let mut var_dict: Vec<T> = Vec::new();
        var_dict.resize(self.n_var as usize, T::zero());
        var_dict[self.one_var as usize] = T::one();
        var_dict
    }
}

impl Index<TensorAddress> for MemoryManager {
    type Output = VariableTensor;
    fn index(&self, idx: TensorAddress) -> &Self::Output {
        &self.mem_dict[idx as usize]
    }
}

pub struct ConstraintSystem {
    a: Vec<(u32, u32, BigScalar)>,
    b: Vec<(u32, u32, BigScalar)>,
    c: Vec<(u32, u32, BigScalar)>,
    n_cons: u32,
    pub mem: MemoryManager,
    compute: Vec<(Box<[u32]>, Functions)>
}

#[derive(Copy, Clone)]
enum Functions {
    Sum = 0,
    Mul = 1,
    Decompose = 2,
    Relu = 3,
    MaxPool = 4,
    Packing = 5,
    ConvCompact = 6
}

impl ConstraintSystem {
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
                    *data = new_index(*data);
                }
            }
        }
    }

    pub fn get_spartan_instance(&self) -> (libspartan::Instance, usize, usize, usize, usize) {
        fn parse_matrix(mat: &[(u32, u32, BigScalar)]) -> Vec<(usize, usize, [u8; 32])> {
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

    pub fn verify(&self, var_dict: &Memory<BigScalar>) -> bool {
        let (mut ai, mut bi, mut ci) = (0, 0, 0);
        for i in 0..self.n_cons {
            let (mut sa, mut sb, mut sc) = (BigScalar::zero(), BigScalar::zero(), BigScalar::zero());
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

    pub fn load_memory<T: Scalar>(&self, tensor: TensorAddress, var_dict: &mut Memory<T>, data: &Memory<T>) {
        assert_eq!(self.mem[tensor].size(), data.len() as u32);
        for (pos, &data) in izip!(self.mem[tensor].iter(), data) {
            var_dict[pos as usize] = data;
        }
    }

    pub fn compute<T: Scalar>(&self, var_dict: &mut Memory<T>) {
        for (params, func) in self.compute.iter() {
            T::FUNCTIONS[*func as usize](&self.mem, &params, var_dict);
        }
    }

    pub fn cons_size(&self) -> u32 {
        return self.n_cons;
    }

    fn run_mul<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (a, b, res) = (param[0], param[1], param[2]);
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

        self.compute.push((Box::new([a, b, res]), Functions::Mul));
    }

    fn run_sum<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let mut res: T = if param.len() == 3 {var_dict[param[2] as usize]} else {T::zero()};
        for x in mem[param[0]].iter() {
            res += var_dict[x as usize];
        }
        var_dict[param[1] as usize] = res;
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

        let mut params = vec![inp, out];
        if let Some(x) = init {
            params.push(x);
        }
        self.compute.push((params.into_boxed_slice(), Functions::Sum));
    }

    pub fn dot(&mut self, a: TensorAddress,  b: TensorAddress, output: ScalarAddress, bias: Option<ScalarAddress>) {
        let tmp = self.mem.alloc(&[min(self.mem[a].size(),self.mem[b].size())]);
        self.mul(a, b, tmp);
        self.sum(tmp, output, bias);
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
                    let cur_bias = if let Some((b, scale)) = bias {
                        Some(self.mem[b].at_idx(&[layer, i/scale, j/scale]))
                    } else {
                        None
                    };
                    self.dot(cur_input[i as usize][j as usize],cur_weight[layer as usize],self.mem[output].at_idx(&[layer, i, j]),cur_bias)
                }
            }
        }
    }

    fn run_decompose<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (input, output, sign, abs) = (param[0], param[1], param[2], param[3]);
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

            let positive = x.is_nonneg();

            var_dict[sign as usize] = if positive {T::one()} else {-T::one()};
            var_dict[abs as usize] = if positive {x} else {-x};

            let data = var_dict[abs as usize].to_bytes();
            for (i, bit) in bits.iter().enumerate() {
                var_dict[bit as usize] = T::from_i32(((data[i/8] >> (i%8)) & 1) as i32);
            }
        }
    }

    // input should have shape with sign, abs, and output should have one more dimension with length bit size
    pub fn bit_decomposition(&mut self, input: TensorAddress, output: TensorAddress, sign: TensorAddress, abs: TensorAddress, compute: bool) {
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
                self.a.push((sum_cons, bit, BigScalar::from(pow)));
                pow *= 2;

                // bit only 0 or 1 <=> bit^2 = bit
                self.a.push((self.n_cons, bit, BigScalar::one()));
                self.b.push((self.n_cons, bit, BigScalar::one()));
                self.c.push((self.n_cons, bit, BigScalar::one()));
                self.n_cons += 1;
            }
        }

        if compute {
            let params = Box::new([input, output, sign, abs]);
            self.compute.push((params, Functions::Decompose));
        }
    }

    pub fn sign(&mut self, input: TensorAddress, output: TensorAddress, max_bits: u8) {
        let mut dim = self.mem[input].dim.to_vec();
        let abs = self.mem.alloc(&dim);
        dim.push(max_bits as u32);
        let bits = self.mem.alloc(&dim);
        self.bit_decomposition(input, bits, output, abs, true);
    }

    fn run_relu<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        for (input, output) in izip!(mem[param[0]].iter(), mem[param[1]].iter()) {
            let x = var_dict[input as usize];
            var_dict[output as usize] = if x.is_nonneg() {x} else {T::zero()}; // hic hic here
        }
    }

    pub fn relu(&mut self, input: TensorAddress, output: TensorAddress, max_bits: u8) {
        let mut dim = self.mem[input].dim.to_vec();
        let abs = self.mem.alloc(&dim);
        let sign = self.mem.alloc(&dim);
        dim.push(max_bits as u32);
        let bits = self.mem.alloc(&dim);
        self.bit_decomposition(input, bits, sign, abs, true);

        for (abs, input, output) in izip!(self.mem[abs].iter(), self.mem[input].iter(), self.mem[output].iter()) {
            self.a.push((self.n_cons, input, BigScalar::one()));
            self.a.push((self.n_cons, abs, BigScalar::one()));
            self.a.push((self.n_cons, output, BigScalar::from(2u32)));
            self.n_cons += 1;
        }

        let params = Box::new([input, output]);
        self.compute.push((params, Functions::Relu));
    }

    fn run_max_pool<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (input, output, temp) = (param[0], param[1], param[2]);
        for layer in 0..mem[input].dim[0] {
            let input = mem[input].at_(&[layer]);
            let output = mem[output].at_(&[layer]);
            for i in 0..input.dim[0]/2 {
                for j in 0..input.dim[1]/2 {
                    let t = [mem[temp].at_idx(&[layer, i, j, 0]), mem[temp].at_idx(&[layer, i, j, 1])];
                    let mut val = [T::zero(),T::zero()];
                    for k in 0..2 {
                        val[k] = if var_dict[input.at_idx(&[2*i + k as u32, 2*j]) as usize] == T::one() ||
                                    var_dict[input.at_idx(&[2*i + k as u32, 2*j + 1]) as usize] == T::one() {
                            T::zero()
                        } else {
                            T::from_i32(2)
                        };
                        var_dict[t[k] as usize] = val[k];
                    }
                    var_dict[output.at_idx(&[i,j]) as usize] = if val[0] == T::zero() || val[1] == T::zero() {
                        T::one()
                    } else {
                        -T::one()
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
                        self.a.push((self.n_cons, input.at_idx(&[2*i + k, 2*j]), BigScalar::one()));
                        self.a.push((self.n_cons, self.mem.one_var, -BigScalar::one()));
                        self.b.push((self.n_cons, input.at_idx(&[2*i + k, 2*j + 1]), BigScalar::one()));
                        self.b.push((self.n_cons, self.mem.one_var, -BigScalar::one()));
                        self.c.push((self.n_cons, t[k as usize], BigScalar::from(2u32)));
                        self.n_cons += 1;
                    }
                    self.a.push((self.n_cons, t[0], BigScalar::one()));
                    self.b.push((self.n_cons, t[1], BigScalar::one()));
                    self.c.push((self.n_cons,  output.at_idx(&[i,j]), -BigScalar::from(2u32)));
                    self.c.push((self.n_cons,  self.mem.one_var, BigScalar::from(2u32)));
                    self.n_cons += 1;
                }
            }
        }

        let params = Box::new([input, output, temp]);
        self.compute.push((params, Functions::MaxPool));
    }

    pub fn fully_connected(&mut self, input: TensorAddress, output: TensorAddress, weight: TensorAddress, bias: Option<TensorAddress>) {
        for i in 0..self.mem[weight].dim[0] {
            let weight = self.mem.save(self.mem[weight].at_(&[i]));
            match bias {
                Some(b) => self.dot(weight,input, self.mem[output].at_idx(&[i]), Some(self.mem[b].at_idx(&[i]))),
                None => self.dot(weight,input, self.mem[output].at_idx(&[i]), None)
            }
        }
    }

    pub fn run_packing_tensor<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let mut in_iter = mem[param[0]].iter();
        let mut out_iter = mem[param[1]].iter();
        let (bit_length, n_packed) = (param[2] & 255, param[2] >> 8);
        let scale = param[3];
        let offset = if param.len() == 12 {
            T::slice_u32_to_scalar(&param[4..12])
        } else {
            T::one()
        };

        let base =  T::from_i32(1 << bit_length);
        let mut is_running = true;
        let mut step = scale;
        while is_running {
            let res = if let Some(r) = out_iter.next() {r} else {break};
            let mut pow = T::one();
            let mut sum_pow = T::zero();
            let mut sum_res = T::zero();
            for i in 0..n_packed {
                step -= 1;
                sum_pow += pow;
                pow *= base;
                if step == 0 || i == n_packed - 1 {
                    if let Some(var) = in_iter.next() {
                        step = scale;
                        sum_res += var_dict[var as usize] * sum_pow;
                        sum_pow = T::zero();
                    } else {
                        is_running = false;
                        break;
                    }
                }
            }
            var_dict[res as usize] = sum_res * offset;
        }
    }

    pub fn packing_tensor(&mut self, input: TensorAddress, output: TensorAddress, bit_length: u8, n_packed: u8, scale: u32, offset: BigScalar, compute: bool) {
        let mut in_iter = self.mem[input].iter();
        let mut out_iter = self.mem[output].iter();

        let base =  BigScalar::from(1u32 << bit_length);
        let mut is_running = true;
        let mut step = scale;
        while is_running {
            let res = if let Some(r) = out_iter.next() {r} else {break};
            let mut pow: BigScalar = BigScalar::one();
            let mut sum_pow = BigScalar::zero();
            for i in 0..n_packed {
                step -= 1;
                sum_pow += pow;
                pow *= base;
                if step == 0 || i == n_packed - 1 {
                    if let Some(var) = in_iter.next() {
                        step = scale;
                        self.a.push((self.n_cons, var, sum_pow * offset));
                        sum_pow = BigScalar::zero();
                    } else {
                        is_running = false;
                        break;
                    }
                }
            }

            self.b.push((self.n_cons, self.mem.one_var, Scalar::one()));
            self.c.push((self.n_cons, res, Scalar::one()));
            self.n_cons += 1;
        }
        if compute {
            let mut params = vec![input, output, bit_length as u32 + ((n_packed as u32) << 8), scale];
            if offset != BigScalar::one() {
                params.extend_from_slice(&scalar_to_vec_u32(offset));
            }
            self.compute.push((params.into_boxed_slice(), Functions::Packing));
        }
    }

    pub fn run_conv2d_compact<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (mul_result, output,k_col, packed_size,bit_length,extracted) = (param[0], param[1], param[2], param[3], param[4], param[5]);
        let (fout, row_out, col_packed) = (mem[mul_result].dim[0],mem[mul_result].dim[1],mem[mul_result].dim[2]);

        let mut offset = T::one();
        for _ in 0..bit_length - 1 {
            offset *= T::from_i32(2);
        }
        let mut big_offset = T::zero();
        for _ in 0..packed_size + k_col - 1 {
            big_offset = (big_offset * T::from_i32(2) + T::one()) * offset;
        }

        for layer_out in 0..fout {
            //matching result
            for r in 0..row_out {
                for c in 0..col_packed {
                    let val = (var_dict[mem[mul_result].at_idx(&[layer_out, r, c]) as usize] + big_offset).to_bytes();
                    let mut ext = Vec::new();
                    ext.resize((packed_size + k_col - 1) as usize, T::zero());
                    for k in 0..(packed_size + k_col - 1) * bit_length {
                        ext[(k / bit_length) as usize] += T::from_i32((((val[(k/8) as usize] >> (k % 8)) & 1) as i32) << (k % bit_length));
                    }
                    for k in 0..packed_size + k_col - 1 {
                        var_dict[mem[extracted].at_idx(&[layer_out,r,c,k]) as usize] = ext[k as usize] - offset;
                    }
                }
            }
        }

        for layer_out in 0..fout {
            //matching result
            for r in 0..row_out {
                for c in 0..col_packed {
                    for k in k_col - 1..packed_size {
                        let rc = (k - (k_col - 1)) + c * packed_size;
                        if rc >= mem[output].dim[2] {break};
                        //correct result for rc
                        let mut res = T::zero();
                        res += var_dict[mem[extracted].at_idx(&[layer_out,r,c,k]) as usize];
                        var_dict[mem[output].at_idx(&[layer_out,r,rc]) as usize] = res;
                    }
                    if c < col_packed - 1 {
                        for k in packed_size..packed_size + k_col - 1 {
                            let rc = (k - (k_col - 1)) + c * packed_size;
                            if rc >= mem[output].dim[2] {break};
                            //correct result for rc
                            let mut res = T::zero();
                            res += var_dict[mem[extracted].at_idx(&[layer_out,r,c,k]) as usize];
                            res += var_dict[mem[extracted].at_idx(&[layer_out,r,c + 1,k - packed_size]) as usize];

                            var_dict[mem[output].at_idx(&[layer_out,r,rc]) as usize] = res;
                        }
                    }
                }
            }
        }
    }

    pub fn conv2d_compact(&mut self, input: TensorAddress, output: TensorAddress, weight_rev: TensorAddress, bias: Option<(TensorAddress, u32)>, bit_length: u8) {
        // packing weight
        let dim = &self.mem[weight_rev].dim;
        let (fout, fin, k_row, k_col) = (dim[0], dim[1], dim[2], dim[3]);

        let packed_weight = self.mem.alloc(&[fout, fin, k_row]);
        assert!(k_col * (bit_length as u32) <= SCALAR_SIZE);
        self.packing_tensor(weight_rev, packed_weight, bit_length, k_col as u8,1, BigScalar::one(), true);

        let (row, col) = (self.mem[input].dim[1], self.mem[input].dim[2]);
        let packed_size = min((SCALAR_SIZE / (bit_length as u32)).checked_sub(k_col).unwrap(),col);
        let col_packed = (col-1)/packed_size + 1;
        let packed_layer = self.mem.alloc(&[fin, row, col_packed]);

        //packing row of inputs
        for layer in 0..fin {
            for r in 0..row {
                let input_row = self.mem.save(self.mem[input].at_(&[layer, r]));
                let packed_row = self.mem.save(self.mem[packed_layer].at_(&[layer, r]));
                self.packing_tensor(input_row, packed_row, bit_length, packed_size as u8,1, BigScalar::one(), true);
            }
        }

        // splicing output by row
        let mut mul_input = Vec::new();
        for r in 0..row - k_row + 1 {
            let mut mul_input_row = Vec::new();
            for c in 0..col_packed {
                mul_input_row.push(self.mem.save(self.mem[packed_layer].at(&[RangeFull(), Range(r..r + k_row), Id(c)])));
            }
            mul_input.push(mul_input_row);
        }

        let mul_result = self.mem.alloc(&[fout, row - k_row + 1, col_packed]);
        for layer_out in 0..fout {
            let packed_weight = self.mem.save(self.mem[packed_weight].at_(&[layer_out]));
            for r in 0..row - k_row + 1 {
                // packing bias
                if let Some((b, scale)) = bias {
                    let bias_dim = (col - k_col)/packed_size + 1;
                    let packed_bias = self.mem.alloc(&[bias_dim]);
                    let bias_row = self.mem.save(self.mem[b].at_(&[layer_out, r]));
                    self.packing_tensor(bias_row, packed_bias, bit_length, packed_size as u8, scale,power_of_two(bit_length as u32 * (k_col - 1)), true);
                    for c in 0..col_packed {
                        let cur_bias = if c < bias_dim {Some(self.mem[packed_bias].at_idx(&[c]))} else {None};
                        self.dot(mul_input[r as usize][c as usize], packed_weight, self.mem[mul_result].at_idx(&[layer_out, r, c]), cur_bias);
                    }
                } else {
                    for c in 0..col_packed {
                        self.dot(mul_input[r as usize][c as usize], packed_weight, self.mem[mul_result].at_idx(&[layer_out, r, c]), None);
                    }
                }
            }
        }

        //sign extraction
        let n_packed = packed_size + k_col - 1;
        let extracted = self.mem.alloc(&[fout, row - k_row + 1, col_packed, n_packed]);
        let extracted_bits = self.mem.alloc(&[fout, row - k_row + 1, col_packed, n_packed, bit_length as u32 - 1]);
        let extracted_sign = self.mem.alloc(&[fout, row - k_row + 1, col_packed, n_packed]);
        let extracted_abs = self.mem.alloc(&[fout, row - k_row + 1, col_packed, n_packed]);

        self.packing_tensor(extracted, mul_result, bit_length, n_packed as u8,1,BigScalar::one(), false);

        for layer_out in 0..fout {
            //matching result
            for r in 0..row - k_row + 1 {
                for c in 0..col_packed {
                    for k in k_col - 1..packed_size {
                        let rc = (k - (k_col - 1)) + c * packed_size;
                        if rc >= col - k_col + 1 {break};
                        //correct result for rc
                        self.a.push((self.n_cons, self.mem[extracted].at_idx(&[layer_out,r,c,k]), Scalar::one()));
                        self.b.push((self.n_cons, self.mem.one_var, Scalar::one()));
                        self.c.push((self.n_cons, self.mem[output].at_idx(&[layer_out,r,rc]), Scalar::one()));
                        self.n_cons += 1;
                    }
                    if c < col_packed - 1 {
                        for k in packed_size..packed_size + k_col - 1 {
                            let rc = (k - (k_col - 1)) + c * packed_size;
                            if rc >= col - k_col + 1 {break};
                            self.a.push((self.n_cons, self.mem[extracted].at_idx(&[layer_out,r,c,k]), Scalar::one()));
                            self.a.push((self.n_cons, self.mem[extracted].at_idx(&[layer_out,r,c+1,k - packed_size]), Scalar::one()));
                            self.b.push((self.n_cons, self.mem.one_var, Scalar::one()));
                            self.c.push((self.n_cons, self.mem[output].at_idx(&[layer_out,r,rc]), Scalar::one()));
                            self.n_cons += 1;
                        }
                    }
                }
            }
        }
        let mut params = vec![mul_result, output, k_col, packed_size, bit_length as u32, extracted];
        self.compute.push((params.into_boxed_slice(), Functions::ConvCompact));
        self.bit_decomposition(extracted, extracted_bits, extracted_sign, extracted_abs, true);
    }
}

struct PoseidonHash {

}

#[cfg(test)]
mod tests {
    use core::slice;

    use super::*;

    #[test]
    fn mul_cons_test() {
        let mut x = ConstraintSystem::new();
        let a = x.mem.alloc(&[3]);
        let b = x.mem.alloc(&[3]);
        let c = x.mem.alloc(&[3]);
        x.mul(a, b, c);

        let mut mem: Vec<BigScalar> = slice_to_scalar(&[1,5,2,3,1,6,3,0,0,0]);
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

        let mut mem: Vec<BigScalar> = slice_to_scalar(&[1,5,2,3,-2,0]);
        x.compute(&mut mem);
        assert_eq!(mem[5], BigScalar::from(8u32));
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

        let mut mem: Vec<BigScalar> = slice_to_scalar(&[1,0,1,-1,0,0,0,-2,4,-1,-4,0,3,-4,0,0,0,1,-1,1,-4,2,3,-1,0,-4,2,2,-3,-1,-1,1,2,-1,1,4,4,2,3,-3,0,3,-2,3,0,2,3,3,-2,2,4,3,3,-4,-4,-1,3,1,4,-2,-2,0,-2,4,-3,0,0,0,-2,0,0,0,0,3,4,-3,-4,-1,-1,-4,3,1,-2,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,0,-3,0,1,-4,-1,2,0,0,-4,2,1,3,2,-3,4,-3]);
        mem.resize(x.mem.n_var as usize, Scalar::zero());

        x.compute(&mut mem);
        assert_eq!(mem[87..87+18], slice_to_scalar(&[32,3,-36,-27,-9,59,44,-21,-16,-23,25,-4,-24,-8,21,-15,-33,-1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

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


    #[test]
    fn conv2d_compact_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,5,5]);
        let weight = x.mem.alloc(&[2,2,3,3]);
        let output = x.mem.alloc(&[2,3,3]);
        let bias = x.mem.alloc(&[2,3,3]);


        let weight_rev = x.mem.save(x.mem[weight].reverse(3));

        x.conv2d_compact(input, output, weight_rev, Some((bias, 1)), 7);

        let mut mem: Vec<BigScalar> = slice_to_scalar(&[1,0,1,-1,0,0,0,-2,4,-1,-4,0,3,-4,0,0,0,1,-1,1,-4,2,3,-1,0,-4,2,2,-3,-1,-1,1,2,-1,1,4,4,2,3,-3,0,3,-2,3,0,2,3,3,-2,2,4,3,3,-4,-4,-1,3,1,4,-2,-2,0,-2,4,-3,0,0,0,-2,0,0,0,0,3,4,-3,-4,-1,-1,-4,3,1,-2,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,0,-3,0,1,-4,-1,2,0,0,-4,2,1,3,2,-3,4,-3]);
        mem.resize(x.mem.n_var as usize, Scalar::zero());

        x.compute(&mut mem);
        assert_eq!(mem[87..87+18], slice_to_scalar(&[32,3,-36,-27,-9,59,44,-21,-16,-23,25,-4,-24,-8,21,-15,-33,-1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn conv2d_compact_test_small() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[1,4,3]);
        let weight = x.mem.alloc(&[1,1,3,3]);
        let output = x.mem.alloc(&[1,2,1]);


        let weight_rev = x.mem.save(x.mem[weight].reverse(3));

        x.conv2d_compact(input, output, weight_rev, None, 5);
        let mut mem = x.mem.new_memory::<BigScalar>();
        x.load_memory(input, &mut mem, &slice_to_scalar(&[1,1,2, 1,2,1, 1,1,1, 1,2,1]));
        x.load_memory(weight, &mut mem, &slice_to_scalar(&[1,1,-1, 1,-1,1, 1,1,1]));

        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize..x.mem[output].end() as usize], slice_to_scalar(&[3,7]));
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

        x.load_memory(input, &mut mem,  &slice_to_scalar(&[2,-2,4,3,1]));
        x.load_memory(weight, &mut mem, &slice_to_scalar(&[-2,3,-2,5,3,-1,5,0,3,2]));
        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize .. x.mem[output].end() as usize], slice_to_scalar(&[0, -1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
}