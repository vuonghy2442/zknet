use std::io;
use std::io::Write;
use rand;
use crate::nn;
use crate::nn::NeuralNetwork;
use crate::zk;
use crate::scalar::{slice_to_scalar, to_vec_i32};
use curve25519_dalek::scalar::Scalar;

fn print_result(result: Vec<Scalar>, hash: Vec<Scalar>, truth: u8) {
    println!("Hash value:");
    for r in hash {
        println!("{:#?}", r);
    }
    let result = to_vec_i32(&result);
    println!("Correct result?: {}", if result[0] == 1 {"yes"} else {"no"});
    println!("Ground truth: {}", truth);
}

fn do_zk_proof(network: NeuralNetwork, memory: &[Scalar]) {
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

pub fn zknet_accuracy() {
    let mut rng = rand::thread_rng();
    let network = nn::NeuralNetwork::new(true);
    let mut memory = network.load_weight::<Scalar>("params/params.pkl");
    let (dataset, truth) = nn::load_dataset("dataset");
    print!("Done loading! Enter sample id: ");
    io::stdout().flush().unwrap();
    let mut x: String = String::new();
    io::stdin().read_line(&mut x).expect("Failed to get console input");
    let x = x.trim().parse::<usize>().expect("Failed to parse int");

    let commit_open = Scalar::random(&mut rng);
    println!("Generate random commit open: {:#?}", commit_open);

    let (result, hash) = network.run(&mut memory, &slice_to_scalar(&dataset[x]), Some(Scalar::from(truth[x])), &[commit_open], true);

    print_result(result, hash, truth[x]);
    do_zk_proof(network, &memory);
}