mod tensor;
mod r1cs;
mod nn;

fn main() {
    let network = nn::NeuralNetwork::new();
    let mut memory = network.load_weight("params/params.pkl");
    let dataset = nn::load_dataset("dataset/test.pkl");
    println!("Done loading!");
    let result = network.run(&mut memory, &dataset[0], true);
    for r in result{
        print!("{} ", r);
    }
}
