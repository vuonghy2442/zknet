use std::convert::TryInto;
use std::io;
use std::io::Write;
use rand;
use rand::CryptoRng;
use rand::RngCore;
use crate::nn;
use crate::nn::AccuracyCommitment;
use crate::nn::NeuralNetwork;
use crate::r1cs::elliptic_curve;
use crate::r1cs::elliptic_curve::elliptic_add;
use crate::r1cs::elliptic_curve::elliptic_mul;
use crate::zk;
use crate::scalar::{slice_to_scalar};
use curve25519_dalek::scalar::Scalar;

fn print_result(result: Vec<Vec<Scalar>>, hash: Vec<Vec<Scalar>>, open: &[Scalar], q: &[Scalar]) {
    let param_a: Scalar = elliptic_curve::get_a();
    let param_d: Scalar = elliptic_curve::get_d();
    println!("Hash value:");
    for r in 1..hash.len() {
        assert_eq!(hash[r], hash[r-1]);
    }

    for r in &hash[0] {
        println!("{:#?}", r);
    }

    let mut sum = [Scalar::zero(), Scalar::one()];
    for r in &result {
        // println!("Result");
        // for rr in r {
        //     println!("{:#?}", rr);
        // }
        sum = elliptic_add(&sum, &r, param_a, param_d);
    }

    let mut cur_sum:[Scalar;2] = open.try_into().unwrap();
    for i in 0..=result.len() {
        if cur_sum == sum {
            println!("Correct {}/{} samples", i, result.len());
            break;
        }
        cur_sum = elliptic_add(&cur_sum, q, param_a, param_d);
    }
}

fn do_zk_proof(network: &NeuralNetwork, memory: &[Scalar]) {
    print!("Proof type (nizk/snark/none): ");
    io::stdout().flush().unwrap();
    let mut x: String = String::new();
    io::stdin().read_line(&mut x).expect("Failed to get console input");

    match x.trim(){
        "nizk" => zk::prove_nizk(&network, &memory),
        "snark" => zk::prove_zk_snark(&network, &memory),
        _ => {}
    }
}

fn generate_result_open<R: RngCore + CryptoRng>(rng: &mut R) -> Scalar {
    let limit: Scalar = elliptic_curve::get_order();
    let mut bytes : [u8; 32] = [0; 32];
    'outer: loop {
        rng.fill_bytes(&mut bytes);
        bytes[31] &= 1;
        for (&a, &b) in bytes.iter().rev().zip(limit.as_bytes().iter().rev()) {
            if a < b {
                break 'outer;
            } else if a > b {
                break;
            }
        }
    }
    return Scalar::from_bits(bytes);
}

fn get_p() -> [Scalar; 2] {
    [
        Scalar::from_bits([165, 69, 15, 204, 207, 113, 207, 38, 62, 63, 78, 98, 124, 5, 127, 19, 227, 172, 104, 57, 76, 114, 16, 216, 22, 108, 66, 159, 246, 205, 84, 4] ),
        Scalar::from_bits([117, 164, 25, 122, 240, 125, 11, 247, 5, 194, 218, 37, 43, 92, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
    ]
}

fn get_q() -> [Scalar; 2] {
    [
        Scalar::from_bits([51, 59, 110, 175, 184, 47, 184, 91, 247, 115, 26, 208, 61, 83, 176, 187, 183, 121, 183, 199, 159, 158, 189, 96, 22, 101, 210, 239, 43, 42, 68, 1]  ),
        Scalar::from_bits([195, 126, 111, 187, 143, 117, 76, 228, 128, 168, 7, 204, 30, 130, 136, 48, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 14] )
    ]
}

pub fn zknet_accuracy(verify: bool) {
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
    println!("Generate random commit open (private): {:#?}", commit_open);

    let p = get_p();
    let q = get_q();

    let mut open: [Scalar; 2] = [Scalar::zero(), Scalar::one()];
    let mut all_result: Vec<Vec<Scalar>> = Vec::new();
    let mut all_hash: Vec<Vec<Scalar>> = Vec::new();
    let param_a = elliptic_curve::get_a();
    let param_d = elliptic_curve::get_d();

    for i in 0..x {
        println!("Run for sample {}", i);
        let result_open = generate_result_open(&mut rng); //private
        let cur_open = elliptic_mul(&p, result_open, param_a, param_d);
        open = elliptic_curve::elliptic_add( &open, &cur_open, param_a, param_d);
        let acc = AccuracyCommitment::<Scalar> {
            ground_truth: Scalar::from(truth[i]),
            result_open,
            p,
            q
        };

        let (result, hash) = network.run(&mut memory, &slice_to_scalar(&dataset[i]), Some(acc), &[commit_open], verify);
        all_result.push(result);
        all_hash.push(hash);
        do_zk_proof(&network, &memory);
    }
    print_result(all_result, all_hash, &open, &q);
}