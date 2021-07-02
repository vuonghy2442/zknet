use std::path::Path;

use crate::nn::{NeuralNetwork, AccuracyCommitment};
use crate::nn;
use crate::scalar::slice_to_scalar;
use crate::r1cs::elliptic_curve;
use crate::io;
use curve25519_dalek::scalar::Scalar;
use rand;
use rand::{CryptoRng, RngCore};

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

fn sum_open(a: Scalar, b:Scalar) -> Scalar {
    let r = a + b;
    let limit: Scalar = elliptic_curve::get_order();
    for (&a, &b) in r.as_bytes().iter().rev().zip(limit.as_bytes().iter().rev()) {
        if a < b {
            return r
        } else if a > b {
            break;
        }
    }
    r - limit
}

fn get_p() -> [Scalar; 2] {
    [
        Scalar::from(4u8),
        Scalar::from_bits([76, 130, 153, 225, 227, 248, 189, 252, 230, 93, 83, 140, 31, 24, 25, 166, 160, 41, 87, 122, 154, 76, 139, 222, 19, 171, 116, 205, 184, 155, 121, 5] )
    ]
}

fn get_q() -> [Scalar; 2] {
    [
        Scalar::from(7u8),
        Scalar::from_bits([52, 39, 232, 82, 140, 80, 24, 107, 219, 65, 100, 122, 127, 191, 128, 165, 35, 98, 218, 95, 46, 2, 26, 39, 193, 121, 112, 91, 77, 236, 112, 6])
    ]
}

fn save_memory( memory: &[Scalar], num_vars: usize, save_path: &Path, name: &str, compress_level: i32) {
    io::save_scalar_array(&memory[..num_vars], save_path.join(format!("{}_witness", name)).to_str().unwrap(), compress_level).unwrap();
    io::save_scalar_array(&memory[num_vars + 1..], save_path.join(format!("{}_io", name)).to_str().unwrap(), compress_level).unwrap();
}

pub fn run_acc(network: &NeuralNetwork, memory: &mut [Scalar],  commit_open: Scalar, dataset: Vec<Vec<i32>>, truth: Vec<u8>, id: &[usize], verify: bool, save_path: &str, compress_level: i32) {
    let save_path = Path::new(save_path);
    let mut rng = rand::thread_rng();

    let p = get_p();
    let q = get_q();

    let mut open: [Scalar; 2] = [Scalar::zero(), Scalar::one()];
    let mut all_result: Vec<Vec<Scalar>> = Vec::new();
    let mut all_hash: Vec<Vec<Scalar>> = Vec::new();
    let param_a = elliptic_curve::get_a();
    let param_d = elliptic_curve::get_d();

    let mut agg_open = Scalar::zero(); //public
    let mut total_correct: u32 = 0;
    for &i in id {
        info!("Run for sample {}", i);
        let result_open = generate_result_open(&mut rng); //private
        agg_open = sum_open(agg_open, result_open);
        let cur_open = elliptic_curve::elliptic_mul(&p, result_open, param_a, param_d);
        open = elliptic_curve::elliptic_add( &open, &cur_open, param_a, param_d);
        let acc = AccuracyCommitment::<Scalar> {
            ground_truth: Scalar::from(truth[i]),
            result_open,
            p,
            q
        };

        let (result, hash) = network.run(memory, &slice_to_scalar(&dataset[i]), Some(acc), &[commit_open], verify);
        total_correct += (result[0] == Scalar::one()) as u32;

        let start = quanta::Instant::now();
        save_memory(memory, network.cons.get_num_vars(),save_path, format!("sample_{}", i).as_str(), compress_level);
        let dur = quanta::Instant::now() - start;
        info!("Done save result in {}", dur.as_secs_f64());

        all_result.push(result);
        all_hash.push(hash);
    }

    if !crate::util::all_same(&all_hash) {
        error!("Hashes are not the same! Bugs!");
        panic!();
    }

    debug!("Hash value:");
    for r in &all_hash[0] {
        debug!("{:?}", r.as_bytes());
    }
    info!("The accuracy is  {}/{}", total_correct, id.len());

    io::save_scalar_array(&all_hash[0], save_path.join("hash").to_str().unwrap(), 0).unwrap();
    io::save_to_file(agg_open, save_path.join("open_accuracy").to_str().unwrap()).unwrap();
    io::save_to_file(total_correct, save_path.join("total_correct").to_str().unwrap()).unwrap();
}


fn softmax(x: &mut [f64]) {
    let mut mean: f64 = 0.0;
    for val in x.iter_mut() {
        mean += *val;
    }
    mean /= x.len() as f64;

    let mut sum = 0f64;
    for val in x.iter_mut() {
        *val = (*val - mean).exp();
        sum += *val;
    }
    for val in x.iter_mut() {
        *val /= sum;
    }
}

fn print_result(result: &[Scalar], truth: u8, scaling: f64) {
    let result = crate::scalar::to_vec_i32(&result);
    let mut prob = Vec::new();
    debug!("raw {:?}", &result);
    for r in result{
        prob.push(r as f64/ scaling);
    }
    softmax(&mut prob);

    println!("Predicted result:");
    for (i,r) in prob.iter().enumerate() {
        println!("Prob {}: {}", i, r);
    }
    println!("Ground truth: {}", truth);
}

pub fn run_infer(network: &NeuralNetwork, memory: &mut [Scalar],  commit_open: Scalar, dataset: Vec<Vec<i32>>, truth: Vec<u8>, id: &[usize], verify: bool, save_path: &str, compress_level: i32) {
    let save_path = Path::new(save_path);

    let mut all_result: Vec<Vec<Scalar>> = Vec::new();
    let mut all_hash: Vec<Vec<Scalar>> = Vec::new();
    for &i in id {
        info!("Run for sample {}", i);

        let (result, hash) = network.run(memory, &slice_to_scalar(&dataset[i]), None, &[commit_open], verify);
        print_result(&result, truth[i], network.scaling);

        let start = quanta::Instant::now();
        save_memory(memory, network.cons.get_num_vars(), save_path, format!("sample_{}", i).as_str(), compress_level);
        let dur = quanta::Instant::now() - start;
        info!("Done save result in {}", dur.as_secs_f64());

        all_result.push(result);
        all_hash.push(hash);
    }

    for r in 1..all_hash.len() {
        assert_eq!(all_hash[r], all_hash[r-1]);
    }

    debug!("Hash value:");
    for r in &all_hash[0] {
        debug!("{:?}", r.as_bytes());
    }

    io::save_scalar_array(&all_hash[0], save_path.join("hash").to_str().unwrap(), 0).unwrap();
}

impl NeuralNetwork {
    pub fn run_dataset(&self, commit_open: Scalar, weight: &str, dataset: &str, id: &[usize], verify: bool, save_path: &str, compress_level: i32) {
        let start = quanta::Instant::now();
        let mut memory = self.load_weight::<Scalar>(weight);
        let dur = quanta::Instant::now() - start;
        info!("Done loading weight in {}", dur.as_secs_f64());
        let start = quanta::Instant::now();
        let (dataset, truth) = nn::load_dataset(dataset);
        let dur = quanta::Instant::now() - start;
        info!("Done loading dataset in {}", dur.as_secs_f64());
        if let Some(_) = self.acc {
            run_acc(self, &mut memory, commit_open, dataset, truth, id, verify, save_path, compress_level)
        } else {
            run_infer(self, &mut memory, commit_open, dataset, truth, id, verify, save_path, compress_level)
        }
    }
}

