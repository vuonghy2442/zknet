#[macro_use] extern crate log;

mod tensor;
mod r1cs;
mod nn;
mod scalar;
mod io;
mod serialize;
pub mod util;

use core::panic;
use std::path::Path;

use itertools::Itertools;
use libspartan::NIZK;
use nn::{NeuralNetworkType};
use clap::{App, Arg, ArgMatches, SubCommand};
use curve25519_dalek::scalar::Scalar;
use r1cs::elliptic_curve;

use crate::{io::load_scalar_array, nn::zk::ProofType, r1cs::elliptic_curve::{elliptic_add, elliptic_mul}};
use simplelog::*;

fn verify(m : &ArgMatches) -> bool {
    let nn = io::zknet_load(m.value_of("CIRCUIT_PATH").unwrap());
    let (inst, gens) = nn.get_nizk_instance();
    let commit_params = nn.get_commit_pq_address();

    info!("{}", if let Some(_) = commit_params {
        "Accuracy mode"
    } else {
        "Infer mode"
    });

    let mut point: [Scalar; 2] = elliptic_curve::get_id();
    let param_a = elliptic_curve::get_a();
    let param_d = elliptic_curve::get_d();
    let proves = nn::zk::get_proves(m.value_of("PROOF_PATH").unwrap(),m.value_of("IO_PATH").unwrap());
    let total_sample = proves.len();
    if total_sample == 0{
        error!("No sample to verify");
        return false;
    }

    let mut p_vec: Vec<[Scalar;2]> = Vec::new();
    let mut q_vec: Vec<[Scalar;2]> = Vec::new();

    for (proof, io, proof_type, id) in proves {
        info!("Verify proof for sample {}", id);
        let inps: Vec<[u8; 32]> = load_scalar_array(&io).unwrap().to_vec();
        if let Some(pos) = &commit_params {
            let [commit, p, q] = &pos;
            point = elliptic_add(&point, &[Scalar::from_bits(inps[commit[0]]), Scalar::from_bits(inps[commit[1]])], param_a, param_d);
            p_vec.push([Scalar::from_bits(inps[p[0]]), Scalar::from_bits(inps[p[1]])]);
            q_vec.push([Scalar::from_bits(inps[q[0]]), Scalar::from_bits(inps[q[1]])]);
        };
        info!("Done load io");
        match proof_type {
            ProofType::NIZK => {
                nn::zk::verify_nizk(&inst, &gens, io::load_from_file::<NIZK>(&proof).unwrap(), &inps);
            }
            _ => {}
        }
    }
    if let Some(_) = commit_params {
        debug!("Reference point {:?}", point);
        if !util::all_same(&p_vec) ||  !util::all_same(&q_vec){
            error!("p and q are not equal across all samples");
            return false;
        }
        let p = p_vec[0];
        let q = q_vec[0];
        let info_path = Path::new(m.value_of("IO_PATH").unwrap());
        let agg_open: Scalar = io::load_from_file( info_path.join("open_accuracy").to_str().unwrap()).unwrap();
        debug!("agg_open {:?}", agg_open.as_bytes());
        let total_correct: u32 = io::load_from_file( info_path.join("total_correct").to_str().unwrap()).unwrap();
        let mul_p = elliptic_mul(&p, agg_open, param_a, param_d);
        debug!("open * p {:?}", mul_p);
        let mul_q = elliptic_mul(&q, Scalar::from(total_correct), param_a, param_d);
        debug!("correct * q {:?}", mul_q);
        let sum_pq = elliptic_add(&mul_p, &mul_q, param_a, param_d);
        debug!("sum p q {:?}", sum_pq);
        if sum_pq == point {
            info!("Accuracy verified {}/{}", total_correct, total_sample);
        } else {
            error!("Accuracy is invalid");
            return false;
        }
    }

    return true;
}

fn main() {
    let matches = App::new("Zero knowledge network").version("0.1").setting(clap::AppSettings::SubcommandRequiredElseHelp)
                            .author("Hy Vuong, Lam Nguyen")
                            .about("Compile binary neural networks to R1CS")
                            .arg(Arg::with_name("verbose").short("v").multiple(true).help("verbosity (number of occurence)"))
                            .subcommand(SubCommand::with_name("generate")
                                    .about("Generate R1CS and computation circuit from example binary neural network")
                                    .arg(Arg::with_name("BNN_TYPE")
                                            .required(true)
                                            .help("Type of the binary neural network to generate (either lenet or nin)"))
                                    .arg(Arg::with_name("OUTPUT")
                                            .required(true)
                                            .help("Output path for R1CS and computation circuit"))
                                    .arg(Arg::with_name("compress").long("compress").short("c")
                                            .default_value("3")
                                            .takes_value(true)
                                            .allow_hyphen_values(true)
                                            .help("Compress level"))
                                    .arg(Arg::with_name("accuracy").short("a").long("accuracy").help("Generate circuit for the zknet accuracy")))
                            .subcommand(SubCommand::with_name("gen_open")
                                    .about("Generate open to commit the network weight from message/random")
                                    .arg(Arg::with_name("OPEN_PATH")
                                            .required(true)
                                            .help("Path to save the open value (private)"))
                                    .arg(Arg::with_name("message")
                                            .long("message").short("m").
                                            takes_value(true).required(false)
                                            .help("The message to generate hash from to generate open")))
                            .subcommand(SubCommand::with_name("run")
                                    .about("Run R1CS circuit and generate witness")
                                    .arg(Arg::with_name("CIRCUIT_PATH")
                                            .required(true)
                                            .help("Path to R1CS circuit"))
                                    .arg(Arg::with_name("WEIGHT_PATH")
                                            .required(true)
                                            .help("Path to neural network weights"))
                                    .arg(Arg::with_name("OPEN_PATH")
                                            .required(true)
                                            .help("Path to open of neural network weights commit"))
                                    .arg(Arg::with_name("DATASET_PATH")
                                            .required(true)
                                            .help("Path to the corresponding dataset"))
                                    .arg(Arg::with_name("SAMPLE_INDEX")
                                            .required(true)
                                            .help("The ids of the sample to run (split by comma)"))
                                    .arg(Arg::with_name("OUTPUT_PATH")
                                            .required(true)
                                            .help("Path to save the output witness and io (folder)"))
                                    .arg(Arg::with_name("compress").long("compress").short("c")
                                            .default_value("3")
                                            .takes_value(true)
                                            .allow_hyphen_values(true)
                                            .help("Compress level"))
                                    .arg(Arg::with_name("verify").long("verify")
                                            .help("Verify the generated value")))
                            .subcommand(SubCommand::with_name("proof")
                                    .about("Generate nizk with Spartan")
                                    .arg(Arg::with_name("CIRCUIT_PATH")
                                            .required(true)
                                            .help("Path to R!CS circuit"))
                                    .arg(Arg::with_name("WITNESS_PATH")
                                            .required(true)
                                            .help("Path to witness and io (folder)"))
                                    .arg(Arg::with_name("PROOF_TYPE")
                                            .required(true)
                                            .help("The proof type to generate (either nizk or snark)"))
                                    .arg(Arg::with_name("OUTPUT_PATH")
                                            .required(true)
                                            .help("Path to output the proof")))
                            .subcommand(SubCommand::with_name("verify")
                                    .about("Verify the generated proof")
                                    .arg(Arg::with_name("CIRCUIT_PATH")
                                            .required(true)
                                            .help("Path to R1CS circuit"))
                                    .arg(Arg::with_name("PROOF_PATH")
                                            .required(true)
                                            .help("Path to the proof result (folder)"))
                                    .arg(Arg::with_name("IO_PATH")
                                            .required(true)
                                            .help("Path to IO path (witness folder)"))).get_matches();

    let level = match matches.occurrences_of("verbose") {
        0 => simplelog::LevelFilter::Warn,
        1 => simplelog::LevelFilter::Info,
        2 => simplelog::LevelFilter::Debug,
        _ => simplelog::LevelFilter::Debug
    };

    simplelog::TermLogger::init(level, Config::default(), TerminalMode::Mixed, simplelog::ColorChoice::Auto).unwrap();


    match matches.subcommand() {
        ("generate", Some(m)) => {
            let nn_type = match m.value_of("BNN_TYPE").unwrap() {
                "lenet" => NeuralNetworkType::LeNet,
                "nin" => NeuralNetworkType::NetworkInNetwork,
                "lenet_unoptimized" => NeuralNetworkType::LeNetUnoptimzied,
                "nin_unoptimized" => NeuralNetworkType::NetworkInNetworkUnoptimzied,
                x => {error!("Unknown neural network type {}", x); panic!()}
            };
            let acc = m.is_present("accuracy");
            let start = quanta::Instant::now();
            let nn = nn::NeuralNetwork::zknet_factory(nn_type, acc);
            let dur = quanta::Instant::now() - start;
            info!("Done generating r1cs in {}", dur.as_secs_f64());

            io::zknet_save(
                &nn,
                m.value_of("OUTPUT").unwrap(),
                m.value_of("compress").unwrap().parse().expect("Compression level is an integer")
            );
        },
        ("gen_open", Some(m)) => {
            let open = io::generate_open(m.value_of("message"));
            io::save_to_file(open, m.value_of("OPEN_PATH").unwrap()).unwrap();
        },
        ("run", Some(m)) => {
            let nn = io::zknet_load(m.value_of("CIRCUIT_PATH").unwrap());
            let open: Scalar = io::load_from_file(m.value_of("OPEN_PATH").unwrap()).unwrap();
            let weight_path: &str = m.value_of("WEIGHT_PATH").unwrap();
            let dataset_path: &str = m.value_of("DATASET_PATH").unwrap();
            let ids: Vec<usize> = m.value_of("SAMPLE_INDEX").unwrap().split(',').map(|x| x.parse::<usize>().unwrap()).collect_vec();
            let output_path: Option<&str> = m.value_of("OUTPUT_PATH");
            let verify = m.is_present("verify");
            let compress: i32 = m.value_of("compress").unwrap().parse().expect("Compression level is an integer");
            nn.run_dataset(open, weight_path, dataset_path, &ids, verify, output_path.unwrap(), compress);
        },
        ("proof", Some(m)) => {
            let nn = io::zknet_load(m.value_of("CIRCUIT_PATH").unwrap());

            let witness_path: &str = m.value_of("WITNESS_PATH").unwrap();
            let output_path: &Path = Path::new(m.value_of("OUTPUT_PATH").unwrap());
            let witnesses = nn::zk::get_witnesses(witness_path);
            match m.value_of("PROOF_TYPE") {
                Some("nizk") => {
                    let (inst,gens) = nn.get_nizk_instance();
                    for (witness, io, id) in witnesses {
                        nn::zk::prove_nizk(&inst, &gens, &witness, &io, id, output_path.join(format!("proof_nizk_{}", id)).to_str().unwrap())
                    }
                },
                Some("snark") => {
                    let (inst, gens, comm, decomm) = nn.get_snark_instance();
                    for (witness, io, id) in witnesses {
                        nn::zk::prove_and_verify_snark(&inst, &gens, &decomm, &comm, &witness, &io, id, output_path.join(format!("proof_snark_{}", id)).to_str().unwrap())
                    }
                },
                _ => {}
            }

        },
        ("verify", Some(m)) => {
            verify(m);
        }
        _ => {}
    }
}
