use std::collections::HashMap;
use super::{NeuralNetwork, ConstraintSystem, convolution_layer, convolution_layer_act_compact, linear_compact, sign_activation, max_pool, AccuracyParams, TensorAddress};
use crate::r1cs::{ActivationFunction, elliptic_curve::{get_a, get_d}};


macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

impl NeuralNetwork {
    pub fn new_lenet(accuracy: bool) -> NeuralNetwork {
        let mut c = ConstraintSystem::new();
        let input = c.mem.alloc(&[1,28,28]);
        let input_cons = c.cons_size();
        let (conv1_out, conv1_weight, conv1_bias) = convolution_layer(&mut c, input, [5,5], 20, 0); // 24 x 24
        let conv1_cons = c.cons_size();
        let conv1_out_sign = sign_activation(&mut c, conv1_out, 22);
        let conv1_sign_cons = c.cons_size();
        let pool1 = max_pool(&mut c, conv1_out_sign); // 12 x 12
        let pool1_cons = c.cons_size();
        let (conv2_out_sign, conv2_weight, conv2_bias) = convolution_layer_act_compact(&mut c, pool1, [5,5], 50, 0, 9, ActivationFunction::Sign); // 8 x 8
        let conv2_cons = c.cons_size();
        let pool2 = max_pool(&mut c, conv2_out_sign);
        let pool2_cons = c.cons_size();
        let (relu_out, _, fc1_weight, fc1_bias) = linear_compact(&mut c, pool2, 500, 15, ActivationFunction::Relu);
        let fc1_cons = c.cons_size();
        let (_, fc2_out, fc2_weight, fc2_bias) = linear_compact(&mut c, relu_out, 10, 20, ActivationFunction::Sign);
        let fc2_cons = c.cons_size();

        let conv1_weight_packed = c.packing_and_check_range(conv1_weight, 13,);
        let conv1_bias_packed = c.packing_and_check_range(conv1_bias, 20,);
        let conv2_weight_packed = c.packing_and_check_range(conv2_weight, 1);
        let conv2_bias_packed = c.packing_and_check_range(conv2_bias, 8,);
        let fc1_weight_packed = c.packing_and_check_range(fc1_weight, 1);
        let fc1_bias_packed = c.packing_and_check_range(fc1_bias, 3,);
        let fc2_weight_packed = c.packing_and_check_range(fc2_weight, 10,);
        let fc2_bias_packed = c.packing_and_check_range(fc2_bias, 11,);
        let packed_cons = c.cons_size();

        let commit_open = c.mem.alloc(&[1]);

        let hash_output = c.poseidon_hash(&[commit_open, conv1_weight_packed, conv1_bias_packed, conv2_weight_packed, conv2_bias_packed
            , fc1_weight_packed, fc1_bias_packed, fc2_weight_packed, fc2_bias_packed]);
        let hash_cons = c.cons_size();

        println!("conv1 constraints {}", conv1_cons - input_cons);
        println!("conv1_sign constraints {}", conv1_sign_cons - conv1_cons);
        println!("pool1 constraints {}", pool1_cons - conv1_sign_cons);
        println!("conv2 constraints {}", conv2_cons - pool1_cons);
        println!("pool2 constraints {}", pool2_cons - conv2_cons);
        println!("fc1 constraints {}", fc1_cons - pool2_cons);
        println!("fc2 constraints {}",fc2_cons - fc1_cons);
        println!("packed constraints {}",packed_cons - fc2_cons);
        println!("hash constraints {}",hash_cons - packed_cons);

        let (output, acc) = if accuracy {
            let p = c.mem.alloc(&[2]);
            let q = c.mem.alloc(&[2]);
            let result_open = c.mem.alloc(&[1]);
            let ground_truth = c.mem.alloc(&[1]);
            let value = c.mem.alloc(&[1]);
            let result = c.mem.alloc(&[1]);
            let mul_p = c.mem.alloc(&[2]);
            let commited_result = c.mem.alloc(&[2]);


            c.multiplexer(fc2_out, c.mem[ground_truth].begin(), c.mem[value].begin());
            c.is_max(fc2_out, c.mem[value].begin(), c.mem[result].begin(), 20);
            c.elliptic_mul(&c.mem[p].to_vec(), c.mem[result_open].begin(), &c.mem[mul_p].to_vec(), get_a(), get_d());
            c.elliptic_add_cond(&c.mem[mul_p].to_vec(), &c.mem[q].to_vec(), &c.mem[commited_result].to_vec(), c.mem[result].begin(), get_a(), get_d());
            println!("accuracy constraints {}", c.cons_size() - hash_cons);

            c.reorder_for_spartan(&[input, ground_truth, commited_result, hash_output, p, q]);
            (result, Some(AccuracyParams{
                ground_truth,
                result_open,
                p,
                q
            }))
        } else {
            c.reorder_for_spartan(&[input, fc2_out, hash_output]);
            (fc2_out, None)
        };
        c.sort_cons();

        println!("Constraints {}", c.cons_size());

        let weight_map: HashMap<String, TensorAddress> = hashmap!{
            String::from("conv1_weight") => conv1_weight,
            String::from("conv1_bias") => conv1_bias,
            String::from("conv2_weight") => conv2_weight,
            String::from("conv2_bias") => conv2_bias,
            String::from("fc1_weight") => fc1_weight,
            String::from("fc1_bias") => fc1_bias,
            String::from("fc2_weight") => fc2_weight,
            String::from("fc2_bias") => fc2_bias
        };

        NeuralNetwork {
            cons: c,
            weight_map,
            input,
            output,
            commit_hash: hash_output,
            commit_open,
            acc,
            scaling: 1024.0
        }
    }
}