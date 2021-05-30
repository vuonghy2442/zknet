mod tensor;
mod r1cs;
use r1cs::ComputationCircuit;
use r1cs::TensorAddress;
use std::cmp::max;

fn ConvolutionLayer(c: &mut ComputationCircuit, input: TensorAddress, kernel: [u32;2], feature: u32) -> (TensorAddress, TensorAddress, TensorAddress) {
    let (row, col) = (c.mem[input].dim[1], c.mem[input].dim[2]);
    let conv_out = c.mem.alloc(&[feature, row - kernel[0] + 1, col - kernel[1] + 1]);
    let conv_weight = c.mem.alloc(&[feature,c.mem[input].dim[0],kernel[0],kernel[1]]);
    let conv_bias = c.mem.alloc(&[feature, 1, 1]);
    c.conv2d(input, conv_out, conv_weight, Some((conv_bias, max(row, col))));
    return (conv_out, conv_weight, conv_bias);
}

fn SignActivation(c: &mut ComputationCircuit, input: TensorAddress, max_bits: u8) -> TensorAddress {
    let output = c.mem.alloc(&c.mem[input].dim.clone());
    c.sign(input, output, max_bits);
    return output;
}

fn MaxPool(c: &mut ComputationCircuit, input: TensorAddress) -> TensorAddress {
    let dim = c.mem[input].dim.clone();
    let output = c.mem.alloc(&[dim[0], dim[1]/2, dim[2]/2]);

    c.binary_max_pool(input, output);
    return output;
}

fn Linear(c: &mut ComputationCircuit, input: TensorAddress, n_feature: u32) -> (TensorAddress, TensorAddress, TensorAddress) {
    let output = c.mem.alloc(&[n_feature]);
    let weight = c.mem.alloc(&[n_feature, c.mem[input].size()]);
    let bias = c.mem.alloc(&[n_feature]);

    c.fully_connected(input, output, weight, Some(bias));
    return (output, weight, bias);
}

fn NeuralNetwork() {
    let mut c = ComputationCircuit::new();
    let input = c.mem.alloc(&[1,28,28]);
    let (conv1_out, conv1_weight, conv1_bias) = ConvolutionLayer(&mut c, input, [5,5], 20);
    let conv1_out_sign = SignActivation(&mut c, conv1_out, 25);
    let (conv2_out, conv2_weight, conv2_bias) = ConvolutionLayer(&mut c, conv1_out_sign, [3,3], 20);
    let conv2_out_sign = SignActivation(&mut c, conv2_out, 9);
    let pool1 = MaxPool(&mut c, conv2_out_sign);
    let (conv3_out, conv3_weight, conv3_bias) = ConvolutionLayer(&mut c, pool1, [3,3], 50);
    let conv3_out_sign = SignActivation(&mut c, conv3_out, 11);
    let pool2 = MaxPool(&mut c, conv3_out_sign);
    let (fc1_out, fc1_weight, fc1_bias) = Linear(&mut c, pool2, 500);
    let (fc2_out, fc2_weight, fc2_bias) = Linear(&mut c, fc1_out, 10);

    println!("Constraints {}", c.cons_size());
}

fn main() {
    NeuralNetwork();
}
