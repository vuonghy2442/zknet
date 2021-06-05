mod tensor;
mod r1cs;
mod nn;
mod zk;
use r1cs::{slice_to_scalar, to_vec_i32};
use curve25519_dalek::scalar::Scalar;

fn softmax(x: &mut [f64]) {
    let mut sum = 0f64;
    for val in x.iter_mut() {
        *val = val.exp();
        sum += *val;
    }
    for val in x.iter_mut() {
        *val /= sum;
    }
}
fn main() {
    let network = nn::NeuralNetwork::new();
    let mut memory = network.load_weight::<Scalar>("params/params.pkl");
    let dataset = nn::load_dataset("dataset/test.pkl");
    println!("Done loading!");
    let result = to_vec_i32(&network.run(&mut memory, &slice_to_scalar(&dataset[1]), false));
    let mut prob = Vec::new();
    for r in result{
        prob.push(r as f64/ 2u32.pow(20) as f64);
    }
    softmax(&mut prob);
    for (i,r) in prob.iter().enumerate() {
        println!("Prob {}: {}", i, r);
    }
    println!("");

    // zk::prove_nizk(network, &memory);
}
