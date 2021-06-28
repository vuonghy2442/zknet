mod tensor;
mod r1cs;
mod nn;
mod zk;
mod scalar;
mod infer;
mod accuracy;

use std::io;
use std::io::Write;

fn main() {
    print!("Choose mode (infer/accuracy): ");
    io::stdout().flush().unwrap();
    let mut x: String = String::new();
    io::stdin().read_line(&mut x).expect("Failed to get console input");

    match x.trim(){
        "infer_nin" => infer::zknet_infer_nin(true),
        "infer" => infer::zknet_infer(true),
        "accuracy" => accuracy::zknet_accuracy(false),
        _ => {}
    }

}
