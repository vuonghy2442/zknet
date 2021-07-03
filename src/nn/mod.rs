use crate::r1cs::ConstraintSystem;
use crate::r1cs::{TensorAddress, ActivationFunction};
use std::cmp::max;
use std::convert::TryInto;
use std::path::Path;
use itertools::Itertools;
use serde_pickle::from_reader;
use std::fs::File;
use crate::scalar::Scalar;
use std::collections::HashMap;

mod lenet;
mod nin;
mod run;
pub mod zk;
mod nn_serialize;

fn convolution_layer_act_compact(c: &mut ConstraintSystem, input: TensorAddress, kernel: [u32;2], feature: u32, bias_scale: u32, max_bits: u8, act: ActivationFunction) -> (TensorAddress, TensorAddress, TensorAddress) {
    let (row, col) = (c.mem[input].dim[1], c.mem[input].dim[2]);
    let (row_out, col_out) = (row - kernel[0] + 1, col - kernel[1] + 1);
    let conv_out = c.mem.alloc(&[feature, row_out, col_out ]);
    let conv_weight = c.mem.alloc(&[feature,c.mem[input].dim[0],kernel[0],kernel[1]]);
    let conv_weight_rev = c.mem.save(c.mem[conv_weight].reverse(3));
    let bias_scale = if bias_scale == 0 {max(row, col)} else {bias_scale};
    let conv_bias = c.mem.alloc(&[feature, (row_out-1)/bias_scale + 1, (col_out-1)/bias_scale + 1]);
    c.conv2d_compact(input, conv_out, conv_weight_rev, Some((conv_bias, bias_scale)), max_bits + 1, act);

    return (conv_out, conv_weight, conv_bias);
}

fn padded_convolution_layer_act_compact(c: &mut ConstraintSystem, input: TensorAddress, kernel: [u32;2], feature: u32, bias_scale: u32, max_bits: u8, act: ActivationFunction) -> (TensorAddress, TensorAddress, TensorAddress) {
    let (row, col) = (c.mem[input].dim[1], c.mem[input].dim[2]);
    let conv_out = c.mem.alloc(&[feature, row, col]);
    let conv_weight = c.mem.alloc(&[feature,c.mem[input].dim[0],kernel[0],kernel[1]]);
    let conv_weight_rev = c.mem.save(c.mem[conv_weight].reverse(3));
    let bias_scale = if bias_scale == 0 {max(row, col)} else {bias_scale};
    let conv_bias = c.mem.alloc(&[feature, (row-1)/bias_scale + 1, (col-1)/bias_scale + 1]);
    c.conv2d_padded_compact(input, conv_out, conv_weight_rev, Some((conv_bias, bias_scale)), max_bits + 1, act);

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

fn padded_convolution_layer(c: &mut ConstraintSystem, input: TensorAddress, kernel: [u32;2], feature: u32, bias_scale: u32) -> (TensorAddress, TensorAddress, TensorAddress) {
    let (row, col) = (c.mem[input].dim[1], c.mem[input].dim[2]);
    let conv_out = c.mem.alloc(&[feature, row, col]);
    let conv_weight = c.mem.alloc(&[feature,c.mem[input].dim[0],kernel[0],kernel[1]]);
    let bias_scale = if bias_scale == 0 {max(row, col)} else {bias_scale};
    let conv_bias = c.mem.alloc(&[feature, (row-1)/bias_scale + 1, (col-1)/bias_scale + 1]);
    c.conv2d_padded(input, conv_out, conv_weight, Some((conv_bias, bias_scale)));

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

fn sum_pool(c: &mut ConstraintSystem, input: TensorAddress, kernel: [u32; 2]) -> TensorAddress {
    let dim = c.mem[input].dim.clone();
    let output = c.mem.alloc(&[dim[0], dim[1]/kernel[0], dim[2]/kernel[1]]);

    c.sum_pool(input, output, kernel);
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

fn linear_compact(c: &mut ConstraintSystem, input: TensorAddress, n_feature: u32, max_bits: u8, act: ActivationFunction) -> (TensorAddress, TensorAddress, TensorAddress, TensorAddress) {
    let output = c.mem.alloc(&[n_feature]);
    let weight = c.mem.alloc(&[n_feature, c.mem[input].size()]);
    let bias = c.mem.alloc(&[n_feature]);

    let res = c.fully_connected_compact(input, output, weight, Some(bias), max_bits + 1, act);
    return (res, output, weight, bias);
}

struct AccuracyParams {
    ground_truth: TensorAddress,
    result_open: TensorAddress,
    p: TensorAddress,
    q: TensorAddress,

    output: TensorAddress
}

pub struct NeuralNetwork {
    cons: ConstraintSystem,
    weight_map: HashMap<String, TensorAddress>,
    input: TensorAddress,
    output: TensorAddress,
    commit_hash: TensorAddress,
    commit_open: TensorAddress,

    acc: Option<AccuracyParams>,
    scaling: f64
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


pub enum NeuralNetworkType {
    LeNet,
    NetworkInNetwork
}


impl NeuralNetwork {
    pub fn get_commit_pq_address(&self) -> Option<[[usize; 2];3]> {
        if let Some(acc) = &self.acc {
            let num_vars =  self.cons.get_num_vars() + 1;
            Some([
                self.cons.mem[acc.output].iter().map(|x|  x as usize - num_vars).collect_vec().try_into().unwrap(),
                self.cons.mem[acc.p].iter().map(|x|  x as usize - num_vars ).collect_vec().try_into().unwrap(),
                self.cons.mem[acc.q].iter().map(|x|  x as usize - num_vars).collect_vec().try_into().unwrap(),
            ])
        } else {
            None
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
        let start_compute = quanta::Instant::now();
        self.cons.compute(var_dict);
        let dur_compute = quanta::Instant::now() - start_compute;
        info!("Done computing in {}", dur_compute.as_secs_f64());

        let mut res = Vec::with_capacity(self.cons.mem[self.output].size() as usize);
        for c in self.cons.mem[self.output].iter() {
            res.push(var_dict[c as usize]);
        };

        let mut hash = Vec::with_capacity(self.cons.mem[self.commit_hash].size() as usize);
        for c in self.cons.mem[self.commit_hash].iter() {
            hash.push(var_dict[c as usize]);
        };

        if verify {
            let start_verify = quanta::Instant::now();
            assert!(self.cons.verify(&T::to_big_scalar(&var_dict)));
            let dur_verify = quanta::Instant::now() - start_verify;
            info!("Verified in {}", dur_verify.as_secs_f64());
        }
        (res, hash)
    }

    pub fn zknet_factory(network: NeuralNetworkType, accuracy: bool) -> NeuralNetwork {
        match network {
            NeuralNetworkType::LeNet => NeuralNetwork::new_lenet(accuracy),
            NeuralNetworkType::NetworkInNetwork => NeuralNetwork::new_nin(accuracy)
        }
    }
}
