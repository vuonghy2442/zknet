#[macro_use] extern crate log;

mod tensor;
mod r1cs;
mod nn;
mod scalar;
mod io;
mod serialize;

use core::panic;
use std::path::Path;

use itertools::Itertools;
use libspartan::NIZK;
use nn::{NeuralNetworkType};
use clap::{Arg, App, SubCommand};
use curve25519_dalek::scalar::Scalar;

use crate::nn::zk::ProofType;
use simplelog::*;

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
                "lenet" => {
                    NeuralNetworkType::LeNet
                },
                "nin" => {
                    NeuralNetworkType::NetworkInNetwork
                }
                x => {error!("Unknown neural network type {}", x); panic!()}
            };
            let acc = m.is_present("accuracy");
            let nn = nn::NeuralNetwork::zknet_factory(nn_type, acc);
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
                        nn::zk::prove_snark(&inst, &gens, &decomm, &witness, &io, id, output_path.join(format!("proof_snark_{}", id)).to_str().unwrap())
                    }
                },
                _ => {}
            }

        },
        ("verify", Some(m)) => {
            let nn = io::zknet_load(m.value_of("CIRCUIT_PATH").unwrap());
            let (inst, gens) = nn.get_nizk_instance();
            let proves = nn::zk::get_proves(m.value_of("PROOF_PATH").unwrap(),m.value_of("IO_PATH").unwrap());
            for (proof, io, proof_type, id) in proves {
                info!("Verify proof for sample {}", id);
                match proof_type {
                    ProofType::NIZK => {
                        nn::zk::verify_nizk(&inst, &gens, io::load_from_file::<NIZK>(&proof).unwrap(), &io);
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}
