
use itertools::Itertools;
use libspartan::{ComputationCommitment, ComputationDecommitment, Instance, NIZK, NIZKGens, SNARK, SNARKGens, VarsAssignment};

use merlin::Transcript;
use std::collections::{HashSet, HashMap};
use std::fs;

use crate::io::{self, load_scalar_array};

use super::NeuralNetwork;

pub enum ProofType {
    NIZK,
    SNARK
}

impl NeuralNetwork {
    pub fn get_nizk_instance(&self) -> (Instance, NIZKGens) {
        let inst = self.cons.get_spartan_instance();
        let gens = NIZKGens::new(self.cons.cons_size() as usize,  self.cons.get_num_vars(), self.cons.get_num_inputs());
        println!("Done encoding nizk");
        (inst, gens)
    }

    pub fn get_snark_instance(&self) -> (Instance, SNARKGens, ComputationCommitment, ComputationDecommitment) {
        let inst = self.cons.get_spartan_instance();
        let gens = SNARKGens::new(self.cons.cons_size() as usize,   self.cons.get_num_vars(), self.cons.get_num_inputs(),self.cons.get_non_zeros());
        let (comm, decomm) = SNARK::encode(&inst, &gens);
        println!("Done encoding snark");
        (inst,gens,comm,decomm)
    }
}

// (witness, io)
pub fn get_witnesses(path: &str) -> Vec<(String, String, usize)> {
    let mut idx: HashSet<usize> =  HashSet::new();
    let mut files_map: HashMap<String,String> = HashMap::new();
    for p in fs::read_dir(path).unwrap() {
        if let Ok(file) = p {
            if file.file_type().unwrap().is_file() {
                let file_name = file.file_name();
                let file_name = file_name.to_str().unwrap();
                let str = file_name.split('_').collect_vec();

                if str[0] == "sample" {
                    if let Ok(id) = str[1].parse::<usize>() {
                        files_map.insert(file_name.to_string(), file.path().to_str().unwrap().to_string());
                        idx.insert(id);
                    }
                }
            }
        }
    };

    let mut res: Vec<(String,String, usize)> = Vec::new();
    for i in idx {
        let f_witness = format!("sample_{}_witness", i);
        let f_io = format!("sample_{}_io", i);
        if let Some(f_witness) = files_map.get(f_witness.as_str()) {
            if let Some(f_io) = files_map.get(f_io.as_str()) {
                res.push((f_witness.clone(), f_io.clone(), i));
            }
        }
    };
    res
}


// (proove, io)
pub fn get_proves(proof_path: &str, io_path: &str) -> Vec<(String, String, ProofType, usize)> {
    let mut io_files_map: HashMap<usize,String> = HashMap::new();
    for p in fs::read_dir(io_path).unwrap() {
        if let Ok(file) = p {
            if file.file_type().unwrap().is_file() {
                let file_name = file.file_name();
                let file_name = file_name.to_str().unwrap();
                let str = file_name.split('_').collect_vec();

                if str[0] == "sample" && str[2] == "io" && str.len() == 3 {
                    if let Ok(id) = str[1].parse::<usize>() {
                        io_files_map.insert(id, file.path().to_str().unwrap().to_string());
                    }
                }
            }
        }
    };

    let mut res: Vec<(String, String, ProofType, usize)> = Vec::new();

    for p in fs::read_dir(proof_path).unwrap() {
        if let Ok(file) = p {
            if !file.file_type().unwrap().is_file() { continue }
            let file_name = file.file_name();
            let file_name = file_name.to_str().unwrap();
            let str = file_name.split('_').collect_vec();
            if str[0] != "proof" { continue };
            if let Ok(id) = str[2].parse::<usize>() {
                if let Some(path) = io_files_map.get(&id) {
                    let proof_type = match str[1] {
                        "nizk" => ProofType::NIZK,
                        "snark" => ProofType::SNARK,
                        _ => continue
                    };

                    res.push((file.path().to_str().unwrap().to_string(), path.to_owned(), proof_type, id));
                }
            }
        }
    };
    res
}

pub fn prove_nizk(inst: &Instance, gens: &NIZKGens, witness_path: &str, io_path: &str, id: usize, save_path: &str) {
    let vars: Vec<[u8; 32]> = load_scalar_array(witness_path).unwrap().to_vec();
    let inps: Vec<[u8; 32]> = load_scalar_array(io_path).unwrap().to_vec();
    let assignment_vars = VarsAssignment::new(&vars).unwrap();
    let assignment_inps = VarsAssignment::new(&inps).unwrap();

    println!("[+] Calculating proof for sample {}", id);
    let mut prover_transcript = Transcript::new(b"nizk_example");
    let proof = NIZK::prove(
        &inst,
        assignment_vars,
        &assignment_inps,
        &gens,
        &mut prover_transcript,
    );

    io::save_to_file(proof, save_path).unwrap();
    println!("[+] Writen proof to {}", save_path);
}

pub fn verify_nizk(inst: &Instance, gens: &NIZKGens, proof: NIZK, io_path: &str) -> bool {
    let inps: Vec<[u8; 32]> = load_scalar_array(io_path).unwrap().to_vec();
    let assignment_inps = VarsAssignment::new(&inps).unwrap();
    let mut verifier_transcript = Transcript::new(b"nizk_example");
    proof.verify(inst, &assignment_inps, &mut verifier_transcript, gens).is_ok()
}

pub fn prove_snark(inst: &Instance, gens: &SNARKGens, decomm: &ComputationDecommitment, witness_path: &str, io_path: &str, id: usize, save_path: &str) {
    let vars: Vec<[u8; 32]> = load_scalar_array(witness_path).unwrap().to_vec();
    let inps: Vec<[u8; 32]> = load_scalar_array(io_path).unwrap().to_vec();
    let assignment_vars = VarsAssignment::new(&vars).unwrap();
    let assignment_inps = VarsAssignment::new(&inps).unwrap();

    println!("[+] Calculating proof for sample {}", id);
    let mut prover_transcript = Transcript::new(b"nizk_example");
    let proof = SNARK::prove(
        &inst,
        decomm,
        assignment_vars,
        &assignment_inps,
        &gens,
        &mut prover_transcript,
    );

    io::save_to_file(proof, save_path).unwrap();
    println!("[+] Writen proof to {}", save_path);
}

pub fn verify_snark(gens: &SNARKGens, comm: &ComputationCommitment, proof: SNARK, io_path: &str) -> bool {
    let inps: Vec<[u8; 32]> = load_scalar_array(io_path).unwrap().to_vec();
    let assignment_inps = VarsAssignment::new(&inps).unwrap();
    let mut verifier_transcript = Transcript::new(b"nizk_example");
    proof.verify(comm, &assignment_inps, &mut verifier_transcript, gens).is_ok()
}