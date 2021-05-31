extern crate curve25519_dalek;

use crate::nn;
use libspartan::{InputsAssignment, NIZKGens, VarsAssignment, NIZK};
use curve25519_dalek::scalar::Scalar as SpartanScalar;

use merlin::Transcript;
use std::fs::File;
use std::io::Write;

pub fn to_bytes(val: i32) -> [u8; 32] {
    if val < 0 {
        (-SpartanScalar::from((-val) as u32)).to_bytes()
    } else {
        SpartanScalar::from(val as u32).to_bytes()
    }
}

pub fn prove(network: nn::NeuralNetwork, memory: &[crate::r1cs::Scalar]) {
    let (inst, num_cons, num_vars, num_inputs) = network.get_spartan_instance();
    let gens = NIZKGens::new(num_cons, num_vars, num_inputs);

    println!("[+] Reading witness");
    let mut vars: Vec<[u8; 32]> = Vec::with_capacity(num_vars);
    let mut inps: Vec<[u8; 32]> = Vec::with_capacity(num_inputs);
    for (i, &val) in memory.iter().enumerate() {
        if i == num_vars {
            assert!(val == 1);
            continue;
        }
        if i < num_vars {
            vars.push(to_bytes(val));
        } else {
            inps.push(to_bytes(val));
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

    let ser = serde_json::to_string(&proof).unwrap();
    println!("[+] Proof size: {}", ser.len());

    println!("[+] Writing proof to proof.dat");
    let mut w = File::create("proof.dat").unwrap();
    writeln!(&mut w, "{}", ser).unwrap();

    let mut verifier_transcript = Transcript::new(b"nizk_example");
    assert!(proof
        .verify(&inst, &assignment_inps, &mut verifier_transcript, &gens)
        .is_ok());

    println!("[+] Finished");
}