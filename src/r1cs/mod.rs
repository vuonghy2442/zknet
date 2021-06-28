
use crate::tensor::{VariableTensor, VariableTensorListIter};
use itertools::izip;
use crate::tensor::TensorIndex::{Id, Range, RangeFull, RangeTo, RangeFrom};
use std::ops::{Index};
use std::cmp::{min, max};
use std::usize;
use curve25519_dalek::scalar::Scalar as BigScalar;
use crate::scalar::{self, SCALAR_SIZE, Scalar, power_of_two, scalar_to_vec_u32};

mod conv2d_compact;
mod conv2d_padded_compact;
mod fully_connected_compact;
mod poseidon;
pub mod elliptic_curve;

type ScalarAddress = u32;
pub type TensorAddress = u32;

type Memory<T> = [T];

pub trait Functional: Sized {
    const FUNCTIONS: [fn(mem: &MemoryManager, &[u32], &mut [Self]); 15];
}

impl<T:Scalar> Functional for T {
    const FUNCTIONS: [fn(mem: &MemoryManager, &[u32], &mut [T]); 15] = [
        ConstraintSystem::run_sum::<T>,
        ConstraintSystem::run_mul::<T>,
        ConstraintSystem::run_decompose::<T>,
        ConstraintSystem::run_relu::<T>,
        ConstraintSystem::run_max_pool::<T>,
        ConstraintSystem::run_packing_tensor::<T>,
        ConstraintSystem::run_conv2d_compact::<T>,
        ConstraintSystem::run_sum_two::<T>,
        ConstraintSystem::run_poseidon_perm_box::<T>,
        ConstraintSystem::run_poseidon_hash::<T>,
        ConstraintSystem::run_fully_connected_compact::<T>,
        ConstraintSystem::run_is_max::<T>,
        ConstraintSystem::run_multiplexer::<T>,
        ConstraintSystem::run_elliptic_mul::<T>,
        ConstraintSystem::run_elliptic_add_cond::<T>,
    ];
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
    ConvCompact = 6,
    SumTwo = 7,
    PoseidonPerm = 8,
    PoseidonHash = 9,
    FCCompact = 10,
    IsMax = 11,
    Multiplexer = 12,
    EllipticMul = 13,
    EllipticAddCond = 14,
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
            let pos = match &r {
                Functions::Sum => 1,
                Functions::IsMax => 4,
                Functions::Multiplexer => 3,
                Functions::EllipticMul => 19,
                Functions::EllipticAddCond => 17,
                _ => continue
            };
            for data in params[pos..].iter_mut() {
                *data = new_index(*data);
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
            let (mut count_a, mut count_b, mut count_c) = (0, 0,0);
            while ai < self.a.len() && self.a[ai].0 == i {
                sa += var_dict[self.a[ai].1 as usize] * self.a[ai].2;
                ai += 1;
                count_a += 1;
            }
            while bi < self.b.len() && self.b[bi].0 == i {
                sb += var_dict[self.b[bi].1 as usize] * self.b[bi].2;
                bi += 1;
                count_b += 1;
            }
            while ci < self.c.len() && self.c[ci].0 == i {
                sc += var_dict[self.c[ci].1 as usize] * self.c[ci].2;
                ci += 1;
                count_c += 1;
            }
            if count_a == 0 || count_b == 0 || count_c == 0 {
                println!("Warning weird condition {}", i);
            }
            if sa * sb != sc {
                println!("Constraint {}", i);
                println!("{} {} {}", self.a[ai-1].1, self.b[bi-1].1, self.c[ci-1].1);
                println!("{} {} {}", sa.to_i32(), sb.to_i32(), sc.to_i32());
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

    fn run_sum_two<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (a, b, res) = (param[0], param[1], param[2]);
        for (x, y, z) in izip!(mem[a].iter(), mem[b].iter(), mem[res].iter()) {
            var_dict[z as usize] = var_dict[x as usize] + var_dict[y as usize];
        }
    }

    pub fn sum_two(&mut self, a: TensorAddress, b: TensorAddress, res: TensorAddress) {
        for (x, y, z) in izip!(self.mem[a].iter(), self.mem[b].iter(), self.mem[res].iter()) {
            self.a.push((self.n_cons, x, Scalar::one()));
            self.a.push((self.n_cons, y, Scalar::one()));
            self.b.push((self.n_cons, self.mem.one_var, Scalar::one()));
            self.c.push((self.n_cons, z, Scalar::one()));
            self.n_cons += 1;
        }

        self.compute.push((Box::new([a, b, res]), Functions::SumTwo));
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

    fn run_is_max<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        if let [input, diff, sign, mul, max_pos, result] = *param {
            for (i,j,k) in izip!(mem[input].iter(), mem[diff].iter(), mem[sign].iter()) {
                var_dict[j as usize] = var_dict[max_pos as usize] - var_dict[i as usize];
                var_dict[k as usize] = if var_dict[j as usize].is_nonneg() {
                    T::one()
                } else {
                    -T::one()
                }
            }

            let mul_tensor = VariableTensorListIter::from_tensor_list(&[mem[mul].clone(), VariableTensor::new_const(result, &[1])]);
            let mut prev = mem.one_var;
            for (i, j) in mem[sign].iter().zip(mul_tensor) {
                var_dict[j as usize] = if var_dict[i as usize] == T::one() && var_dict[prev as usize] == T::one() {
                    T::one()
                } else {
                    T::zero()
                };
                prev = j;
            }
        } else {
            panic!("params don't match");
        }
    }

    pub fn is_max(&mut self, input: TensorAddress, max_pos: ScalarAddress, result: ScalarAddress, max_bits: u8) {
        let diff = self.mem.alloc(&[self.mem[input].size()]);
        let sign = self.mem.alloc(&[self.mem[input].size()]);
        let mul = self.mem.alloc(&[self.mem[input].size() - 1]);

        for (i,j) in self.mem[input].iter().zip(self.mem[diff].iter()) {
            self.a.push((self.n_cons, max_pos, BigScalar::one()));
            self.a.push((self.n_cons, i, -BigScalar::one()));
            self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
            self.c.push((self.n_cons, j, BigScalar::one()));
            self.n_cons += 1;
        }

        let mul_tensor = VariableTensorListIter::from_tensor_list(&[self.mem[mul].clone(), VariableTensor::new_const(result, &[1])]);
        let mut prev = self.mem.one_var;
        for (i, j) in self.mem[sign].iter().zip(mul_tensor) {
            self.a.push((self.n_cons, i, BigScalar::one()));
            self.a.push((self.n_cons, self.mem.one_var, BigScalar::one()));
            self.b.push((self.n_cons, prev, BigScalar::one()));
            self.c.push((self.n_cons, j, BigScalar::from_i32(2)));
            self.n_cons += 1;
            prev = j;
        }

        self.compute.push((Box::new([input, diff, sign, mul, max_pos, result]), Functions::IsMax));
        self.sign(diff, sign, max_bits);
    }

    fn run_multiplexer<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        if let [input,index_bits, tmp, result] = *param {
            let mut start = 0;
            let mut cur_layer = VariableTensorListIter::from_tensor_list(&[mem[input].clone()]);
            let mut size = mem[input].size();

            for bit in mem[index_bits].iter() {
                let half_size = size >> 1;
                let is_odd = (size & 1) == 1;
                let cur_tmp = if size == 2 {
                    VariableTensor::new_const(result, &[1])
                } else {
                    mem[tmp].at(&[Range(start..start + half_size)])
                };
                start += half_size;
                size = (size >> 1) + (size & 1);

                for i in cur_tmp.iter() {
                    let (x, y) = (cur_layer.next().unwrap(), cur_layer.next().unwrap());
                    var_dict[i as usize] = var_dict[y as usize] * var_dict[bit as usize] + var_dict[x as usize] * (T::one() - var_dict[bit as usize]);
                }
                cur_layer = if is_odd {
                    VariableTensorListIter::from_tensor_list(&[cur_tmp, VariableTensor::new_const(cur_layer.next().unwrap(), &[1])])
                } else {
                    VariableTensorListIter::from_tensor_list(&[cur_tmp])
                }
            }
        } else {
            panic!("Params don't match");
        }
    }

    pub fn multiplexer(&mut self, input: TensorAddress, index: ScalarAddress, result: ScalarAddress) {
        let mut size = self.mem[input].size();
        let mut n_bits: u32 = 0;
        let mut tmp_size = 0;
        while size > 1 {
            tmp_size += size >> 1;
            size = (size >> 1) + (size & 1);
            n_bits += 1;
        }
        let index = self.mem.save(VariableTensor::new_const(index, &[1]));
        let index_bits = self.mem.alloc(&[1, n_bits]);
        let index_two_complement = self.mem.alloc(&[1]);
        let index_sign = self.mem.alloc(&[1]);

        self.bit_decomposition(index, index_bits, index_sign, index_two_complement, true );

        let tmp = self.mem.alloc(&[tmp_size - 1]);

        let mut start = 0;
        let mut cur_layer = VariableTensorListIter::from_tensor_list(&[self.mem[input].clone()]);
        size = self.mem[input].size();
        for bit in self.mem[index_bits].iter() {
            let is_odd = (size & 1) == 1;

            let cur_tmp = if size == 2 {
                VariableTensor::new_const(result, &[1])
            } else {
                self.mem[tmp].at(&[Range(start..start + (size >> 1))])
            };

            start += size >> 1;
            size = (size >> 1) + (size & 1);

            for i in cur_tmp.iter() {
                let (x, y) = (cur_layer.next().unwrap(), cur_layer.next().unwrap());
                self.a.push((self.n_cons, x, -BigScalar::one()));
                self.a.push((self.n_cons, y, BigScalar::one()));
                self.b.push((self.n_cons, bit, BigScalar::one()));
                self.c.push((self.n_cons, i, BigScalar::one()));
                self.c.push((self.n_cons, x, -BigScalar::one()));
                self.n_cons += 1;
            }
            cur_layer = if is_odd {
                VariableTensorListIter::from_tensor_list(&[cur_tmp, VariableTensor::new_const(cur_layer.next().unwrap(), &[1])])
            } else {
                VariableTensorListIter::from_tensor_list(&[cur_tmp])
            }
        }
        self.compute.push((Box::new([input, index_bits, tmp, result]), Functions::Multiplexer));
    }

    pub fn conv2d(&mut self, input: TensorAddress, output: TensorAddress, weight: TensorAddress, bias: Option<(TensorAddress, u32)>) {
        let fout = self.mem[weight].dim[0];
        let ki = self.mem[weight].dim[2];
        let kj = self.mem[weight].dim[3];
        let (out_row, out_col) = (self.mem[input].dim[1] - ki + 1, self.mem[input].dim[2] - kj + 1);
        let mut cur_weight: Vec<TensorAddress> = Vec::with_capacity(fout as usize);
        for layer in 0..fout {
            cur_weight.push(self.mem.save(self.mem[weight].at(&[Id(layer)])));
        }

        let mut cur_input: Vec<Vec<TensorAddress>> = Vec::with_capacity(out_row as usize);
        for i in 0..out_row {
            let mut tmp: Vec<TensorAddress> = Vec::with_capacity(out_col as usize);
            for j in 0..out_col {
                tmp.push(self.mem.save(self.mem[input].at(&[RangeFull(), Range(i..i+ki), Range(j..j+kj)])));
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

    pub fn conv2d_padded(&mut self, input: TensorAddress, output: TensorAddress, weight: TensorAddress, bias: Option<(TensorAddress, u32)>) {
        let fout = self.mem[weight].dim[0];
        let ki = self.mem[weight].dim[2];
        let kj = self.mem[weight].dim[3];
        assert!(ki % 2 == 1 && kj % 2 == 1);
        let (out_row, out_col) = (self.mem[input].dim[1], self.mem[input].dim[2]);

        let padi = ki / 2;
        let padj = kj / 2;

        let mut cur_input: Vec<Vec<TensorAddress>> = Vec::with_capacity(out_row as usize);
        for i in 0..out_row {
            let mut tmp: Vec<TensorAddress> = Vec::with_capacity(out_col as usize);
            for j in 0..out_col {
                tmp.push(self.mem.save(self.mem[input].at(&[RangeFull(), Range(i.saturating_sub(padi)..min(out_row,i+padi+1)), Range(j.saturating_sub(padj)..min(out_col,j+padj+1))])));
            }
            cur_input.push(tmp);
        }
        for layer in 0..fout {
            let mut cur_weight: Vec<Vec<TensorAddress>> = Vec::with_capacity(ki as usize);
            for i in (0..ki).rev() {
                let mut tmp: Vec<TensorAddress> = Vec::with_capacity(kj as usize);
                for j in (0..kj).rev() {
                    tmp.push(self.mem.save(self.mem[weight].at(&[Id(layer), RangeFull(), Range(i.saturating_sub(padi)..min(ki,i+padi+1)), Range(j.saturating_sub(padj)..min(kj,j+padj+1))])));
                }
                cur_weight.push(tmp);
            }

            for i in 0..out_row{
                let u = if i < padi {i} else if i >= out_row - padi {i - (out_row - 2*padi - 1)} else {padi} as usize;
                for j in 0..out_col{
                    let cur_bias = if let Some((b, scale)) = bias {
                        Some(self.mem[b].at_idx(&[layer, i/scale, j/scale]))
                    } else {
                        None
                    };
                    let v = if j < padj {j} else if j >= out_col - padj {j - (out_col - 2*padj - 1)} else {padj} as usize;
                    self.dot(cur_input[i as usize][j as usize],cur_weight[u][v],self.mem[output].at_idx(&[layer, i, j]),cur_bias)
                }
            }
        }
    }

    fn run_decompose<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (input, output, sign, two_complement) = (param[0], param[1], param[2], param[3]);
        let mut iter = mem[input].iter();
        loop {
            let x = iter.next();
            if let None = x {
                break
            }
            let x = var_dict[x.unwrap() as usize];

            let idx = &iter.idx;
            let bits = mem[output].at_(idx);
            let bits_cnt = bits.size();
            let sign = mem[sign].at_idx(idx);
            let two_complement = mem[two_complement].at_idx(idx);

            let positive = x.is_nonneg();

            var_dict[sign as usize] = if positive {T::one()} else {-T::one()};
            var_dict[two_complement as usize] = if positive {x} else {power_of_two::<T>(bits_cnt) + x};

            let data = var_dict[two_complement as usize].to_bytes();
            for (i, bit) in bits.iter().enumerate() {
                var_dict[bit as usize] = T::from_i32(((data[i/8] >> (i%8)) & 1) as i32);
            }
        }
    }

    // input should have shape with sign, abs, and output should have one more dimension with length bit size
    pub fn bit_decomposition(&mut self, input: TensorAddress, output: TensorAddress, sign: TensorAddress, two_complement: TensorAddress, compute: bool) {
        let mut iter = self.mem[input].iter();
        loop {
            let x = iter.next();
            if let None = x { break };
            let x = x.unwrap();
            let idx = &iter.idx;

            let bits = self.mem[output].at_(idx);
            let bits_cnt = bits.size();
            let two_complement = self.mem[two_complement].at_idx(idx); //abs(2x+1)
            let sign = self.mem[sign].at_idx(idx);
            // two_comp + x = 2^bits when sign = -1 and = 0 when sign = 1
            self.a.push((self.n_cons, two_complement, BigScalar::one()));
            self.a.push((self.n_cons, x, -BigScalar::one()));
            self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
            self.c.push((self.n_cons, sign, -power_of_two::<BigScalar>(bits_cnt - 1)));
            self.c.push((self.n_cons, self.mem.one_var, power_of_two(bits_cnt - 1)));
            self.n_cons += 1;

            // sign only = +-1 <=> sign^2 == 1
            self.a.push((self.n_cons, sign, Scalar::one()));
            self.b.push((self.n_cons, sign, Scalar::one()));
            self.c.push((self.n_cons, self.mem.one_var, Scalar::one()));
            self.n_cons += 1;

            let sum_cons = self.n_cons;
            self.n_cons += 1;
            self.b.push((sum_cons, self.mem.one_var, BigScalar::one()));
            self.c.push((sum_cons, two_complement, BigScalar::one()));


            let mut pow: BigScalar = BigScalar::one();
            for bit in bits.iter() {
                self.a.push((sum_cons, bit, pow));
                pow *= BigScalar::from_i32(2);

                // bit only 0 or 1 <=> bit^2 = bit
                self.a.push((self.n_cons, bit, BigScalar::one()));
                self.b.push((self.n_cons, bit, BigScalar::one()));
                self.c.push((self.n_cons, bit, BigScalar::one()));
                self.n_cons += 1;
            }
        }

        if compute {
            let params = Box::new([input, output, sign, two_complement]);
            self.compute.push((params, Functions::Decompose));
        }
    }

    pub fn sign(&mut self, input: TensorAddress, output: TensorAddress, max_bits: u8) {
        let mut dim = self.mem[input].dim.to_vec();

        let two_complement = self.mem.alloc(&dim);
        dim.push(max_bits as u32);
        let bits = self.mem.alloc(&dim);
        self.bit_decomposition(input, bits, output, two_complement, true);
    }

    fn run_relu<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        for (input, output) in izip!(mem[param[0]].iter(), mem[param[1]].iter()) {
            let x = var_dict[input as usize];
            var_dict[output as usize] = if x.is_nonneg() {x} else {T::zero()}; // hic hic here
        }
    }

    pub fn relu(&mut self, input: TensorAddress, output: TensorAddress, max_bits: u8) {
        let mut dim = self.mem[input].dim.to_vec();
        let two_complement = self.mem.alloc(&dim);
        let sign = self.mem.alloc(&dim);
        dim.push(max_bits as u32);
        let bits = self.mem.alloc(&dim);
        self.bit_decomposition(input, bits, sign, two_complement, true);

        for (sign, input, output) in izip!(self.mem[sign].iter(), self.mem[input].iter(), self.mem[output].iter()) {
            self.a.push((self.n_cons, input, BigScalar::one()));
            self.b.push((self.n_cons, sign, BigScalar::one()));
            self.b.push((self.n_cons, self.mem.one_var, BigScalar::one()));
            self.c.push((self.n_cons, output, BigScalar::from(2u32)));
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

    pub fn packing_tensor(&mut self, input: TensorAddress, output: TensorAddress, bit_length: u8, n_packed: u8, scale: u32, offset: BigScalar, compute: bool) {
        let mut in_iter = self.mem[input].iter();
        let mut out_iter = self.mem[output].iter();

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
        if compute {
            let mut params = vec![input, output, bit_length as u32 + ((n_packed as u32) << 8), scale];
            if offset != BigScalar::one() {
                params.extend_from_slice(&scalar_to_vec_u32(offset));
            }
            self.compute.push((params.into_boxed_slice(), Functions::Packing));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::slice_to_scalar;

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
        let bias = x.mem.alloc(&[1]);
        let res = x.mem.alloc(&[1]);

        x.sum(a, x.mem[res].begin(), Some(x.mem[bias].begin()));

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
    fn conv2d_padded_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,5,5]);
        let weight = x.mem.alloc(&[2,2,3,3]);
        let output = x.mem.alloc(&[2,5,5]);
        let bias = x.mem.alloc(&[2,5,5]);

        x.conv2d_padded(input, output, weight, Some((bias, 1)));
        let mut mem = x.mem.new_memory();
        x.load_memory(input, &mut mem, &[-1,0,2,0,2,0,1,-1,0,2,1,0,2,1,1,-1,0,1,0,1,-1,0,-1,0,0,1,0,0,0,1,0,2,0,0,1,0,0,2,0,0,0,1,1,0,2,-1,0,-1,2,0]);
        x.load_memory(weight, &mut mem, &[0,-1,1,1,1,2,2,2,1,1,0,0,1,2,-1,0,0,-1,-1,0,-1,1,1,2,0,0,2,1,-1,-1,-1,0,0,1,0,1]);
        x.load_memory(bias, &mut mem, &[1,2,-1,2,0,-1,2,-1,1,2,2,-1,0,-1,2,0,2,2,-1,0,-1,0,1,0,-1,2,0,2,1,0,0,1,2,2,1,-1,-1,-1,0,1,1,-1,0,-1,0,1,0,-1,0,0]);

        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize..x.mem[output].end() as usize], [1,7,1,6,8,2,10,4,12,8,1,-2,13,9,4,-5,4,1,1,4,-3,-2,-4,4,0,5,0,6,12,2,1,6,2,4,2,-2,6,5,6,2,0,-9,-1,-1,2,-1,-4,-2,-3,-4]);
        x.sort_cons();
        let mem = slice_to_scalar(&mem);
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


    #[test]
    fn max_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[5]);
        let max = x.mem.alloc(&[1]);
        let output = x.mem.alloc(&[1]);
        x.is_max(input, x.mem[max].begin(), x.mem[output].begin(), 10);
        x.reorder_for_spartan(&[input]);
        let mut mem = x.mem.new_memory();
        x.load_memory(input, &mut mem,  &slice_to_scalar(&[2,-2,4,3,1]));
        x.load_memory(max, &mut mem, &slice_to_scalar(&[4]));
        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize], BigScalar::from_i32(1));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
    #[test]
    fn max_test_2() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[5]);
        let max = x.mem.alloc(&[1]);
        let output = x.mem.alloc(&[1]);
        x.is_max(input, x.mem[max].begin(), x.mem[output].begin(), 10);
        let mut mem = x.mem.new_memory();
        x.load_memory(input, &mut mem,  &slice_to_scalar(&[2,-2,4,3,1]));
        x.load_memory(max, &mut mem, &slice_to_scalar(&[3]));
        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize], BigScalar::from_i32(0));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn multiplexer_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[5]);
        let index = x.mem.alloc(&[1]);
        let output = x.mem.alloc(&[1]);
        x.multiplexer(input, x.mem[index].begin(), x.mem[output].begin());
        x.reorder_for_spartan(&[input]);
        let mut mem = x.mem.new_memory();
        x.load_memory(input, &mut mem,  &slice_to_scalar(&[1,3,5,7,9]));
        x.load_memory(index, &mut mem, &slice_to_scalar(&[3]));
        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize], BigScalar::from_i32(7));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
}