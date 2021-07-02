use std::{fs::File, io::{BufWriter}};

use bincode::{deserialize_from, serialize_into};
use curve25519_dalek::scalar::Scalar;
use serde::{Serialize, de::DeserializeOwned};
use zstd;
use sha2::Sha512;

use crate::{nn::{NeuralNetwork}, serialize::MySerialize};

pub fn zknet_save(network: &NeuralNetwork, path: &str, compress_level: i32) {
    let f = BufWriter::new(File::create(path).unwrap());
    let mut f_comp = zstd::stream::Encoder::new(f, compress_level).unwrap();
    info!("Save the neural network to {} with compression {}", path, compress_level);
    network.my_serialize(& mut f_comp);
    f_comp.finish().unwrap();
    info!("Done saving the neural network");
}

pub fn zknet_load(path: &str) -> NeuralNetwork  {
    info!("Load neural network from {}", path);
    let f = File::open(path).unwrap();
    let mut f_comp = zstd::stream::Decoder::new(f).unwrap();
    let res = NeuralNetwork::my_deserialize(&mut f_comp);
    info!("Done loading neural network");
    res
}

pub fn generate_open(message: Option<&str>) -> Scalar {
    let res = match message {
        Some(m) => {
            info!("Generate open from message {}", m);
            Scalar::hash_from_bytes::<Sha512>(m.as_bytes())
        }
        None => {
            info!("Generate open randomly");
            let mut rng = rand::thread_rng();
            Scalar::random(&mut rng)
        }
    };
    debug!("Generated open {:?}", res.as_bytes());
    res

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