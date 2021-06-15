mod tensor;
mod r1cs;
mod nn;
mod zk;
mod scalar;
use std::io;
use std::io::Write;

use crate::scalar::{slice_to_scalar, to_vec_i32};
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
    print!("Done loading! Enter sample id: ");
    io::stdout().flush().unwrap();
    let mut x: String = String::new();
    io::stdin().read_line(&mut x).expect("Failed to get console input");
    let x = x.trim().parse::<usize>().expect("Failed to parse int");
    let (result, hash) = network.run(&mut memory, &slice_to_scalar(&dataset[x]), true);
    println!("Hash value:");
    for r in hash {
        println!("{:#?}", r);
    }
    println!("Predicted result:");

    let result = to_vec_i32(&result);
    let mut prob = Vec::new();
    for (i, &r) in result.iter().enumerate(){
        println!("Raw {}: {}", i, r);
        prob.push(r as f64/ 2u32.pow(10) as f64);
    }
    softmax(&mut prob);
    for (i,r) in prob.iter().enumerate() {
        println!("Prob {}: {}", i, r);
    }

    print!("Proof type (nizk/snark/none): ");
    io::stdout().flush().unwrap();
    let mut x: String = String::new();
    io::stdin().read_line(&mut x).expect("Failed to get console input");

    match x.trim(){
        "nizk" => zk::prove_nizk(network, &memory),
        "snark" => zk::prove_zk_snark(network, &memory),
        _ => {}
    }

}
