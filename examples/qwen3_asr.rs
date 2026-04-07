/*
Transcribe wav file using Qwen3-ASR

cargo run --example qwen3_asr -- \
  --conv-frontend path/to/conv_frontend.onnx \
  --encoder path/to/encoder.int8.onnx \
  --decoder path/to/decoder.int8.onnx \
  --tokenizer path/to/tokenizer \
  path/to/test.wav
*/

use sherpa_rs::{
    qwen3_asr::{Qwen3AsrConfig, Qwen3AsrRecognizer},
    read_audio_file,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut conv_frontend = String::new();
    let mut encoder = String::new();
    let mut decoder = String::new();
    let mut tokenizer = String::new();
    let mut wav_path = String::new();
    let mut provider = "cpu".to_string();
    let mut num_threads: i32 = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--conv-frontend" => {
                i += 1;
                conv_frontend = args[i].clone();
            }
            "--encoder" => {
                i += 1;
                encoder = args[i].clone();
            }
            "--decoder" => {
                i += 1;
                decoder = args[i].clone();
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = args[i].clone();
            }
            "--provider" => {
                i += 1;
                provider = args[i].clone();
            }
            "--threads" => {
                i += 1;
                num_threads = args[i].parse().unwrap();
            }
            _ => {
                wav_path = args[i].clone();
            }
        }
        i += 1;
    }

    if conv_frontend.is_empty()
        || encoder.is_empty()
        || decoder.is_empty()
        || tokenizer.is_empty()
        || wav_path.is_empty()
    {
        eprintln!(
            "Usage: {} --conv-frontend <conv_frontend.onnx> --encoder <encoder.onnx> --decoder <decoder.onnx> --tokenizer <tokenizer_dir> [--provider cpu] <wav_file>",
            args[0]
        );
        std::process::exit(1);
    }

    let (samples, sample_rate) = read_audio_file(&wav_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    println!("Provider: {}, Threads: {}", provider, num_threads);
    let config = Qwen3AsrConfig {
        conv_frontend,
        encoder,
        decoder,
        tokenizer,
        provider: Some(provider),
        num_threads: Some(num_threads),
        debug: false,
        ..Default::default()
    };

    let mut recognizer = Qwen3AsrRecognizer::new(config).unwrap();

    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    let elapsed = start_t.elapsed();
    let audio_duration = samples.len() as f32 / sample_rate as f32;
    let rtf = elapsed.as_secs_f32() / audio_duration;

    println!("Text: {}", result.text);
    println!("Time taken: {:.3}s", elapsed.as_secs_f32());
    println!("Audio duration: {:.3}s", audio_duration);
    println!("RTF: {:.3}", rtf);
}
