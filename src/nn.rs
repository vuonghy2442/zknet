use crate::r1cs::ConstraintSystem;
use crate::r1cs::{TensorAddress};
use std::cmp::max;
use std::path::Path;
use serde_pickle::from_reader;
use std::fs::File;
use std::collections::HashMap;
use crate::scalar::Scalar;
use crate::r1cs::elliptic_curve::{get_a, get_d};

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

fn convolution_layer_sign_compact(c: &mut ConstraintSystem, input: TensorAddress, kernel: [u32;2], feature: u32, bias_scale: u32, max_bits: u8) -> (TensorAddress, TensorAddress, TensorAddress) {
    let (row, col) = (c.mem[input].dim[1], c.mem[input].dim[2]);
    let (row_out, col_out) = (row - kernel[0] + 1, col - kernel[1] + 1);
    let conv_out = c.mem.alloc(&[feature, row_out, col_out ]);
    let conv_weight = c.mem.alloc(&[feature,c.mem[input].dim[0],kernel[0],kernel[1]]);
    let conv_weight_rev = c.mem.save(c.mem[conv_weight].reverse(3));
    let bias_scale = if bias_scale == 0 {max(row, col)} else {bias_scale};
    let conv_bias = c.mem.alloc(&[feature, (row_out-1)/bias_scale + 1, (col_out-1)/bias_scale + 1]);
    c.conv2d_compact(input, conv_out, conv_weight_rev, Some((conv_bias, bias_scale)), max_bits + 1);

    return (conv_out, conv_weight, conv_bias);
}

fn convolution_layer(c: &mut ConstraintSystem, input: TensorAddress, kernel: [u32;2], feature: u32, bias_scale: u32) -> (TensorAddress, TensorAddress, TensorAddress) {
    let (row, col) = (c.mem[input].dim[1], c.mem[input].dim[2]);
    let (row_out, col_out) = (row - kernel[0] + 1, col - kernel[1] + 1);
    let conv_out = c.mem.alloc(&[feature, row_out, col_out ]);
    let conv_weight = c.mem.alloc(&[feature,c.mem[input].dim[0],kernel[0],kernel[1]]);
    let bias_scale = if bias_scale == 0 {max(row, col)} else {bias_scale};
    let conv_bias = c.mem.alloc(&[feature, (row_out-1)/bias_scale + 1, (col_out-1)/bias_scale + 1]);
    c.conv2d(input, conv_out, conv_weight, Some((conv_bias, bias_scale)));

    return (conv_out, conv_weight, conv_bias);
}

fn sign_activation(c: &mut ConstraintSystem, input: TensorAddress, max_bits: u8) -> TensorAddress {
    let output = c.mem.alloc(&c.mem[input].dim.clone());
    c.sign(input, output, max_bits);
    return output;
}

fn max_pool(c: &mut ConstraintSystem, input: TensorAddress) -> TensorAddress {
    let dim = c.mem[input].dim.clone();
    let output = c.mem.alloc(&[dim[0], dim[1]/2, dim[2]/2]);

    c.binary_max_pool(input, output);
    return output;
}

fn resize(c: &mut ConstraintSystem, input: TensorAddress, shape: &[u32]) -> TensorAddress {
    c.mem.save(c.mem[input].resize(shape))
}

fn relu_activation(c: &mut ConstraintSystem, input: TensorAddress, max_bits: u8) -> TensorAddress {
    let output = c.mem.alloc(&c.mem[input].dim.clone());
    c.relu(input, output, max_bits);
    return output;
}

fn linear(c: &mut ConstraintSystem, input: TensorAddress, n_feature: u32) -> (TensorAddress, TensorAddress, TensorAddress) {
    let output = c.mem.alloc(&[n_feature]);
    let weight = c.mem.alloc(&[n_feature, c.mem[input].size()]);
    let bias = c.mem.alloc(&[n_feature]);

    c.fully_connected(input, output, weight, Some(bias));
    return (output, weight, bias);
}

fn linear_compact(c: &mut ConstraintSystem, input: TensorAddress, n_feature: u32, max_bits: u8, relu: bool) -> (TensorAddress, TensorAddress, TensorAddress) {
    let output = c.mem.alloc(&[n_feature]);
    let weight = c.mem.alloc(&[n_feature, c.mem[input].size()]);
    let bias = c.mem.alloc(&[n_feature]);

    let res = c.fully_connected_compact(input, output, weight, Some(bias), max_bits + 1, relu);
    return (if relu {res} else {output}, weight, bias);
}

struct AccuracyParams {
    ground_truth: TensorAddress,
    result_open: TensorAddress,
    p: TensorAddress,
    q: TensorAddress
}

pub struct NeuralNetwork {
    cons: ConstraintSystem,
    weight_map: HashMap<String, TensorAddress>,
    input: TensorAddress,
    output: TensorAddress,
    commit_hash: TensorAddress,
    commit_open: TensorAddress,

    acc: Option<AccuracyParams>
}

pub struct AccuracyCommitment<T: Scalar> {
    pub ground_truth: T,
    pub result_open: T,
    pub p: [T;2],
    pub q: [T;2],
}

pub fn load_dataset(path: &str) -> (Vec<Vec<i32>>, Vec<u8>) {
    let path = Path::new(path);
    let f = File::open(path.join("test.pkl")).unwrap();
    let data: Vec<Vec<i32>>= from_reader(f).unwrap();
    let f = File::open(path.join("truth.pkl")).unwrap();
    let truth: Vec<u8>= from_reader(f).unwrap();
    (data, truth)
}


impl NeuralNetwork {
    pub fn new(accuracy: bool) -> NeuralNetwork {
        let mut c = ConstraintSystem::new();
        let input = c.mem.alloc(&[1,28,28]);
        let input_cons = c.cons_size();
        let (conv1_out, conv1_weight, conv1_bias) = convolution_layer(&mut c, input, [5,5], 20, 0); // 24 x 24
        let conv1_cons = c.cons_size();
        let conv1_out_sign = sign_activation(&mut c, conv1_out, 22);
        let conv1_sign_cons = c.cons_size();
        let pool1 = max_pool(&mut c, conv1_out_sign); // 12 x 12
        let pool1_cons = c.cons_size();
        let (conv2_out_sign, conv2_weight, conv2_bias) = convolution_layer_sign_compact(&mut c, pool1, [5,5], 50, 0, 9); // 8 x 8
        let conv2_cons = c.cons_size();
        let pool2 = max_pool(&mut c, conv2_out_sign);
        let pool2_cons = c.cons_size();
        let (relu_out, fc1_weight, fc1_bias) = linear_compact(&mut c, pool2, 500, 15, true);
        let fc1_cons = c.cons_size();
        let (fc2_out, fc2_weight, fc2_bias) = linear_compact(&mut c, relu_out, 10, 20, false);
        let fc2_cons = c.cons_size();

        let conv1_weight_packed = c.packing_and_check_range(conv1_weight, 13, false);
        let conv1_bias_packed = c.packing_and_check_range(conv1_bias, 20, false);
        let conv2_weight_packed = c.packing_and_check_range(conv2_weight, 1, true);
        let conv2_bias_packed = c.packing_and_check_range(conv2_bias, 8, false);
        let fc1_weight_packed = c.packing_and_check_range(fc1_weight, 1, true);
        let fc1_bias_packed = c.packing_and_check_range(fc1_bias, 3, false);
        let fc2_weight_packed = c.packing_and_check_range(fc2_weight, 10, false);
        let fc2_bias_packed = c.packing_and_check_range(fc2_bias, 11, false);
        let packed_cons = c.cons_size();

        let commit_open = c.mem.alloc(&[1]);

        let hash_output = c.poseidon_hash(&[commit_open, conv1_weight_packed, conv1_bias_packed, conv2_weight_packed, conv2_bias_packed
            , fc1_weight_packed, fc1_bias_packed, fc2_weight_packed, fc2_bias_packed]);
        let hash_cons = c.cons_size();

        println!("conv1 constraints {}", conv1_cons - input_cons);
        println!("conv1_sign constraints {}", conv1_sign_cons - conv1_cons);
        println!("pool1 constraints {}", pool1_cons - conv1_sign_cons);
        println!("conv2 constraints {}", conv2_cons - pool1_cons);
        println!("pool2 constraints {}", pool2_cons - conv2_cons);
        println!("fc1 constraints {}", fc1_cons - pool2_cons);
        println!("fc2 constraints {}",fc2_cons - fc1_cons);
        println!("packed constraints {}",packed_cons - fc2_cons);
        println!("hash constraints {}",hash_cons - packed_cons);

        let (output, acc) = if accuracy {
            let p = c.mem.alloc(&[2]);
            let q = c.mem.alloc(&[2]);
            let result_open = c.mem.alloc(&[1]);
            let ground_truth = c.mem.alloc(&[1]);
            let value = c.mem.alloc(&[1]);
            let result = c.mem.alloc(&[1]);
            let mul_p = c.mem.alloc(&[2]);
            let commited_result = c.mem.alloc(&[2]);


            c.multiplexer(fc2_out, c.mem[ground_truth].begin(), c.mem[value].begin());
            c.is_max(fc2_out, c.mem[value].begin(), c.mem[result].begin(), 20);
            c.elliptic_mul(&c.mem[p].to_vec(), c.mem[result_open].begin(), &c.mem[mul_p].to_vec(), get_a(), get_d());
            c.elliptic_add_cond(&c.mem[mul_p].to_vec(), &c.mem[q].to_vec(), &c.mem[commited_result].to_vec(), c.mem[result].begin(), get_a(), get_d());
            println!("accuracy constraints {}", c.cons_size() - hash_cons);

            c.reorder_for_spartan(&[input, ground_truth, commited_result, hash_output, p, q]);
            (commited_result, Some(AccuracyParams{
                ground_truth,
                result_open,
                p,
                q
            }))
        } else {
            c.reorder_for_spartan(&[input, fc2_out, hash_output]);
            (fc2_out, None)
        };
        c.sort_cons();

        println!("Constraints {}", c.cons_size());

        let weight_map: HashMap<String, TensorAddress> = hashmap!{
            String::from("conv1_weight") => conv1_weight,
            String::from("conv1_bias") => conv1_bias,
            String::from("conv2_weight") => conv2_weight,
            String::from("conv2_bias") => conv2_bias,
            String::from("fc1_weight") => fc1_weight,
            String::from("fc1_bias") => fc1_bias,
            String::from("fc2_weight") => fc2_weight,
            String::from("fc2_bias") => fc2_bias
        };

        NeuralNetwork {
            cons: c,
            weight_map,
            input,
            output,
            commit_hash: hash_output,
            commit_open,
            acc
        }
    }

    pub fn load_weight<T: Scalar>(&self, file: &str) -> Vec<T> {
        let w = File::open(file).unwrap();
        let weight: HashMap<String, Vec<i32>>= from_reader(w).unwrap();

        let mut memory = self.cons.mem.new_memory::<T>();
        for (key, address) in self.weight_map.iter() {
            let mut data = Vec::new();
            for &x in weight[key].iter() {data.push(T::from_i32(x));};
            self.cons.load_memory(*address, &mut memory, &data);
        };
        memory
    }

    pub fn get_spartan_instance(&self) -> (libspartan::Instance, usize, usize, usize, usize) {
        return self.cons.get_spartan_instance();
    }

    pub fn run<T: Scalar>(&self, var_dict: &mut [T], input: &[T], accuracy_commitment: Option<AccuracyCommitment<T>>, commit_open: &[T], verify: bool) -> (Vec<T>, Vec<T>) {
        self.cons.load_memory(self.input, var_dict, input);

        if let Some(t) = accuracy_commitment {
            let acc = self.acc.as_ref().unwrap();
            self.cons.load_memory(acc.ground_truth, var_dict, &[t.ground_truth]);
            self.cons.load_memory(acc.result_open, var_dict, &[t.result_open]);
            self.cons.load_memory(acc.p, var_dict, &t.p);
            self.cons.load_memory(acc.q, var_dict, &t.q);
        } else {
            assert!(matches!(self.acc, None));
        }

        self.cons.load_memory(self.commit_open, var_dict, commit_open);
        self.cons.compute(var_dict);

        println!("Done compute");

        let mut res = Vec::with_capacity(self.cons.mem[self.output].size() as usize);
        for c in self.cons.mem[self.output].iter() {
            res.push(var_dict[c as usize]);
        };

        let mut hash = Vec::with_capacity(self.cons.mem[self.commit_hash].size() as usize);
        for c in self.cons.mem[self.commit_hash].iter() {
            hash.push(var_dict[c as usize]);
        };

        if verify {
            assert!(self.cons.verify(&T::to_big_scalar(&var_dict)));
            println!("Verified");
        }
        (res, hash)
    }
}