mod tensor;
mod r1cs;
mod nn;
mod zk;
use r1cs::slice_to_scalar;

fn main() {
    let network = nn::NeuralNetwork::new();
    let mut memory = network.load_weight("params/params.pkl");
    let dataset = nn::load_dataset("dataset/test.pkl");
    println!("Done loading!");
    let result = network.run(&mut memory, &slice_to_scalar(&dataset[0]), true);
//     // for r in result{
//     //     print!("{} ", String::from(r));
//     // }
//     println!("");

    zk::prove_nizk(network, &mut memory);
}
