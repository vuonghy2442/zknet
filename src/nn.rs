use crate::r1cs::ConstraintSystem;
use crate::r1cs::TensorAddress;
use crate::r1cs::Scalar;
use std::cmp::max;
use serde_pickle::from_reader;
use std::fs::File;
use std::collections::HashMap;

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
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
    output: TensorAddress
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
        let (conv2_out, conv2_weight, conv2_bias) = convolution_layer(&mut c, conv1_out_sign, [3,3], 20, 0);
        let conv2_out_sign = sign_activation(&mut c, conv2_out, 9);
        let pool1 = max_pool(&mut c, conv2_out_sign);
        let (conv3_out, conv3_weight, conv3_bias) = convolution_layer(&mut c, pool1, [3,3], 50, 2);
        let conv3_out_sign = sign_activation(&mut c, conv3_out, 11);
        let pool2 = max_pool(&mut c, conv3_out_sign);
        let (fc1_out, fc1_weight, fc1_bias) = linear(&mut c, pool2, 500);
        let relu_out = relu_activation(&mut c, fc1_out, 11);
        let (fc2_out, fc2_weight, fc2_bias) = linear(&mut c, relu_out, 10);
        c.reorder_for_spartan(&[input, fc2_out]);
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
            output: fc2_out
        }
    }

    pub fn load_weight(&self, file: &str) -> Vec<Scalar> {
        let w = File::open(file).unwrap();
        let weight: HashMap<String, Vec<i32>>= from_reader(w).unwrap();

        let mut memory = self.cons.mem.new_memory();
        for (key, address) in self.weight_map.iter() {
            self.cons.load_memory(*address, &mut memory, &weight[key])
        };
        memory
    }

    pub fn get_spartan_instance(&self) -> (libspartan::Instance, usize, usize, usize) {
        return self.cons.get_spartan_instance();
    }

    pub fn run(&self, var_dict: &mut [Scalar], input: &[Scalar], verify: bool) -> Vec<Scalar> {
        self.cons.load_memory(self.input, var_dict, input);
        self.cons.compute(var_dict);

        let mut res = Vec::with_capacity(self.cons.mem[self.output].size() as usize);
        for c in self.cons.mem[self.output].iter() {
            res.push(var_dict[c as usize]);
        };
        if verify {
            assert!(self.cons.verify(&var_dict));
        }
        res
    }
}