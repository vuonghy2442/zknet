extern crate curve25519_dalek;

use crate::nn;
use libspartan::{InputsAssignment, NIZKGens, VarsAssignment, NIZK, SNARKGens, SNARK};

use merlin::Transcript;
use std::fs::File;
use std::io::Write;
use curve25519_dalek::scalar::Scalar;

pub fn prove_nizk(network: nn::NeuralNetwork, memory: &[Scalar]) {
    let (inst, num_cons, num_vars, num_inputs,_) = network.get_spartan_instance();
    let gens = NIZKGens::new(num_cons, num_vars, num_inputs);

    let mut vars: Vec<[u8; 32]> = Vec::with_capacity(num_vars);
    let mut inps: Vec<[u8; 32]> = Vec::with_capacity(num_inputs);
    for (i, &val) in memory.iter().enumerate() {
        if i == num_vars {
            assert!(val == Scalar::one());
            continue;
        }
        if i < num_vars {
            vars.push(val.to_bytes());
        } else {
            inps.push(val.to_bytes());
        }
    }
    let assignment_vars = VarsAssignment::new(&vars).unwrap();
    let assignment_inps = InputsAssignment::new(&inps).unwrap();

    println!("[+] Calculating proof");
    let mut prover_transcript = Transcript::new(b"nizk_example");
    let proof = NIZK::prove(
        &inst,
        assignment_vars,
        &assignment_inps,
        &gens,
        &mut prover_transcript,
    );

    let bin = bincode::serialize(&proof).unwrap();
    println!("[+] Proof size: {}", bin.len());
    println!("[+] Writing proof to proof.dat");
    let mut w = File::create("proof.dat").unwrap();
    w.write_all(&bin).unwrap();

    let mut verifier_transcript = Transcript::new(b"nizk_example");
    assert!(proof
        .verify(&inst, &assignment_inps, &mut verifier_transcript, &gens)
        .is_ok());

    println!("[+] Finished");
}


pub fn prove_zk_snark(network: nn::NeuralNetwork, memory: &[Scalar]) {
    let (inst, num_cons, num_vars, num_inputs,non_zero) = network.get_spartan_instance();
    let gens = SNARKGens::new(num_cons, num_vars, num_inputs,non_zero);
    let (comm, decomm) = SNARK::encode(&inst, &gens);
    println!("[+] encode");
    let mut vars: Vec<[u8; 32]> = Vec::with_capacity(num_vars);
    let mut inps: Vec<[u8; 32]> = Vec::with_capacity(num_inputs);
    for (i, &val) in memory.iter().enumerate() {
        if i == num_vars {
            assert!(val == Scalar::one());
            continue;
        }
        if i < num_vars {
            vars.push(val.to_bytes());
        } else {
            inps.push(val.to_bytes());
        }
    }
    let assignment_vars = VarsAssignment::new(&vars).unwrap();
    let assignment_inps = InputsAssignment::new(&inps).unwrap();

    println!("[+] Calculating proof");
    let mut prover_transcript = Transcript::new(b"nizk_example");
    let proof = SNARK::prove(
        &inst,
        &decomm,
        assignment_vars,
        &assignment_inps,
        &gens,
        &mut prover_transcript,
    );

    let bin = bincode::serialize(&proof).unwrap();
    println!("[+] Proof size: {}", bin.len());
    println!("[+] Writing proof to proof.dat");
    let mut w = File::create("proof.dat").unwrap();
    w.write_all(&bin).unwrap();

    let mut verifier_transcript = Transcript::new(b"nizk_example");
    assert!(proof
        .verify(&comm, &assignment_inps, &mut verifier_transcript, &gens)
        .is_ok());

    println!("[+] Finished");
}