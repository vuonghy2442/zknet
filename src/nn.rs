use crate::r1cs::ConstraintSystem;
use crate::r1cs::{TensorAddress};
use std::cmp::max;
use serde_pickle::from_reader;
use std::fs::File;
use std::collections::HashMap;
use crate::scalar::Scalar;


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

pub struct NeuralNetwork {
    cons: ConstraintSystem,
    weight_map: HashMap<String, TensorAddress>,
    input: TensorAddress,
    output: TensorAddress,
    commit_hash: TensorAddress
}

pub fn load_dataset(file: &str) -> Vec<Vec<i32>> {
    let w = File::open(file).unwrap();
    let data: Vec<Vec<i32>>= from_reader(w).unwrap();
    data
}


impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        let mut c = ConstraintSystem::new();
        let input = c.mem.alloc(&[1,28,28]);
        let input_resized = resize(&mut c, input, &[1, 26, 26]);
        let (conv1_out, conv1_weight, conv1_bias) = convolution_layer(&mut c, input_resized, [5,5], 20, 0);
        let conv1_out_sign = sign_activation(&mut c, conv1_out, 25);
        let (conv2_out_sign, conv2_weight, conv2_bias) = convolution_layer_sign_compact(&mut c, conv1_out_sign, [3,3], 20, 0, 9);
        let pool1 = max_pool(&mut c, conv2_out_sign);
        let (conv3_out_sign, conv3_weight, conv3_bias) = convolution_layer_sign_compact(&mut c, pool1, [3,3], 50, 2, 11);
        let pool2 = max_pool(&mut c, conv3_out_sign);
        let (fc1_out, fc1_weight, fc1_bias) = linear(&mut c, pool2, 500);
        let relu_out = relu_activation(&mut c, fc1_out, 11);
        let (fc2_out, fc2_weight, fc2_bias) = linear(&mut c, relu_out, 10);

        let conv1_weight_packed = c.packing_and_check_range(conv1_weight, 16, false);
        let conv1_bias_packed = c.packing_and_check_range(conv1_bias, 23, false);
        let conv2_weight_packed = c.packing_and_check_range(conv2_weight, 1, true);
        let conv2_bias_packed = c.packing_and_check_range(conv2_bias, 7, false);
        let conv3_weight_packed = c.packing_and_check_range(conv3_weight, 1, true);
        let conv3_bias_packed = c.packing_and_check_range(conv3_bias, 12, false);
        let fc1_weight_packed = c.packing_and_check_range(fc1_weight, 1, true);
        let fc1_bias_packed = c.packing_and_check_range(fc1_bias, 3, false);
        let fc2_weight_packed = c.packing_and_check_range(fc2_weight, 19, false);
        let fc2_bias_packed = c.packing_and_check_range(fc2_bias, 21, false);

        let hash_output = c.poseidon_hash(&[conv1_weight_packed, conv1_bias_packed, conv2_weight_packed, conv2_bias_packed,
            conv3_weight_packed, conv3_bias_packed, fc1_weight_packed, fc1_bias_packed,
            fc2_weight_packed, fc2_bias_packed]);

        c.reorder_for_spartan(&[input, fc2_out, hash_output]);
        c.sort_cons();
        println!("Constraints {}", c.cons_size());

        let weight_map: HashMap<String, TensorAddress> = hashmap!{
            String::from("conv1_weight") => conv1_weight,
            String::from("conv1_bias") => conv1_bias,
            String::from("conv2_weight") => conv2_weight,
            String::from("conv2_bias") => conv2_bias,
            String::from("conv3_weight") => conv3_weight,
            String::from("conv3_bias") => conv3_bias,
            String::from("fc1_weight") => fc1_weight,
            String::from("fc1_bias") => fc1_bias,
            String::from("fc2_weight") => fc2_weight,
            String::from("fc2_bias") => fc2_bias
        };

        NeuralNetwork {
            cons: c,
            weight_map,
            input,
            output: fc2_out,
            commit_hash: hash_output
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

    pub fn run<T: Scalar>(&self, var_dict: &mut [T], input: &[T], verify: bool) -> (Vec<T>, Vec<T>) {
        self.cons.load_memory(self.input, var_dict, input);
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