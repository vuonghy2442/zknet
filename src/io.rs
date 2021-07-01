use std::{fs::File, io::{BufWriter}};

use bincode::{deserialize_from, serialize_into};
use curve25519_dalek::scalar::Scalar;
use serde::{Serialize, de::DeserializeOwned};
use zstd;
use sha2::Sha512;

use crate::nn::{NeuralNetwork};

pub fn zknet_save(network: &NeuralNetwork, path: &str, compress_level: i32) -> bincode::Result<()> {
    let f = BufWriter::new(File::create(path).unwrap());
    let mut f_comp = zstd::stream::Encoder::new(f, compress_level).unwrap();
    let result = serialize_into(&mut f_comp, &network);
    f_comp.finish().unwrap();
    result
}

pub fn zknet_load(path: &str) -> bincode::Result<NeuralNetwork>  {
    let f_comp = zstd::stream::Decoder::new(File::open(path).unwrap()).unwrap();
    bincode::deserialize_from(f_comp)
}

pub fn generate_open(message: Option<&str>) -> Scalar {
    match message {
        Some(m) => {
            Scalar::hash_from_bytes::<Sha512>(m.as_bytes())
        }
        None => {
            let mut rng = rand::thread_rng();
            Scalar::random(&mut rng)
        }
    }
}

pub fn save_to_file<T:Serialize>(obj: T, path: &str) -> bincode::Result<()> {
    let w = File::create(path).unwrap();
    serialize_into(w, &obj)
}


pub fn load_from_file<T:DeserializeOwned>(path: &str) -> bincode::Result<T> {
    let r = File::open(path).unwrap();
    deserialize_from(r)
}

pub fn save_scalar_array<T: Serialize>(arr: &[T], path: &str, compress_level: i32) -> bincode::Result<()> {
    let f = BufWriter::new(File::create(path).unwrap());
    let mut f_comp = zstd::stream::Encoder::new(f, compress_level).unwrap();
    let res = serialize_into(&mut f_comp, arr);
    f_comp.finish().unwrap();
    res
}

pub fn load_scalar_array<T: DeserializeOwned>(path: &str) -> bincode::Result<Box<[T]>> {
    let mut f_comp = zstd::stream::Decoder::new(File::open(path).unwrap()).unwrap();
    deserialize_from(&mut f_comp)
}