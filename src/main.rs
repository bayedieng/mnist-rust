use memmap2::MmapOptions;
use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand::Rng;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;

fn to_ndarray_map(tensors: Vec<(String, TensorView)>) -> HashMap<String, Array2<f32>> {
    let mut arr = Vec::new();
    let mut hash_map: HashMap<String, Array2<f32>> = HashMap::new();
    for (name, tensor) in tensors {
        let chunks = tensor.data().chunks_exact(4);
        for chunk in chunks {
            let chunk_buf: [u8; 4] = chunk.try_into().unwrap();
            let float = f32::from_le_bytes(chunk_buf);
            arr.push(float)
        }

        let shape = if tensor.shape().len() > 1 {
            (tensor.shape()[0], tensor.shape()[1])
        } else {
            (1, tensor.shape()[0])
        };
        let ndarray = Array2::from_shape_vec(shape, arr.clone()).unwrap();
        arr.clear();
        hash_map.insert(name, ndarray);
    }
    hash_map
}

fn softmax(arr: &Array2<f32>) -> Array2<f32> {
    let stable_max = *arr.clone().max().unwrap();
    (arr - stable_max) / arr.mapv(f32::exp).sum()
}

fn main() {
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .base_path("mnist/MNIST/raw")
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.0);
    let train_labels: Array1<f32> = Array1::from_shape_vec(50_000, trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let x = train_data.slice(s![4, 0..]).to_owned();
    run_inference_on_torch_weights(&x);
    let w1: Array2<f32> = Array2::random((784, 128), StandardNormal);
    let w2: Array1<f32> = Array1::random(128, StandardNormal);
}

fn run_inference_on_torch_weights(input_data: &Array1<f32>) {
    let torch_weights_file = File::open("mnist_classifier.safetensors").unwrap();
    let buffer = unsafe { MmapOptions::new().map(&torch_weights_file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();

    let torch_weights = to_ndarray_map(tensors.tensors());
    let l1 = &torch_weights["L1.weight"].dot(input_data);
    let l2 = &torch_weights["L2.weight"].dot(l1);
    let out = l2.argmax().unwrap();
    println!("Image Class is {:?}", out)
}
