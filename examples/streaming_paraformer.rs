/*
Streaming transcription using Paraformer

Models: paraformer-streaming directory with encoder.int8.onnx, decoder.int8.onnx, tokens.txt

cargo run --example streaming_paraformer -- \
  --encoder path/to/encoder.int8.onnx \
  --decoder path/to/decoder.int8.onnx \
  --tokens path/to/tokens.txt \
  path/to/test.wav
*/

use sherpa_rs::{
    read_audio_file,
    streaming_paraformer::{StreamingParaformerConfig, StreamingParaformerRecognizer},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut encoder = String::new();
    let mut decoder = String::new();
    let mut tokens = String::new();
    let mut wav_path = String::new();
    let mut provider = "cpu".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--encoder" => {
                i += 1;
                encoder = args[i].clone();
            }
            "--decoder" => {
                i += 1;
                decoder = args[i].clone();
            }
            "--tokens" => {
                i += 1;
                tokens = args[i].clone();
            }
            "--provider" => {
                i += 1;
                provider = args[i].clone();
            }
            _ => {
                wav_path = args[i].clone();
            }
        }
        i += 1;
    }

    if encoder.is_empty() || decoder.is_empty() || tokens.is_empty() || wav_path.is_empty() {
        eprintln!(
            "Usage: {} --encoder <encoder.onnx> --decoder <decoder.onnx> --tokens <tokens.txt> [--provider cpu] <wav_file>",
            args[0]
        );
        std::process::exit(1);
    }

    let (samples, sample_rate) = read_audio_file(&wav_path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = StreamingParaformerConfig {
        encoder,
        decoder,
        tokens,
        provider: Some(provider),
        debug: true,
        ..Default::default()
    };

    let recognizer = StreamingParaformerRecognizer::new(config).unwrap();
    let stream = recognizer.create_stream().unwrap();

    let start_t = std::time::Instant::now();

    // Simulate streaming: feed audio in chunks (e.g., 100ms chunks at 16kHz = 1600 samples)
    let chunk_size = 1600; // 100ms at 16kHz
    let mut last_text = String::new();

    for chunk in samples.chunks(chunk_size) {
        stream.accept_waveform(sample_rate as i32, chunk);

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }

        let result = recognizer.get_result(&stream);
        if !result.text.is_empty() && result.text != last_text {
            println!("  [partial] {}", result.text);
            last_text = result.text;
        }

        if recognizer.is_endpoint(&stream) {
            let result = recognizer.get_result(&stream);
            if !result.text.is_empty() {
                println!("  [endpoint] {}", result.text);
            }
            recognizer.reset(&stream);
            last_text.clear();
        }
    }

    // Add tail padding (0.3s of silence) to flush remaining tokens
    let tail_padding = vec![0.0f32; (sample_rate as usize) * 3 / 10];
    stream.accept_waveform(sample_rate as i32, &tail_padding);

    // Signal end of audio
    stream.input_finished();

    // Decode remaining
    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }

    let result = recognizer.get_result(&stream);
    if !result.text.is_empty() && result.text != last_text {
        println!("  [partial] {}", result.text);
    }
    println!("\n✅ Final text: {}", result.text);
    println!("⏱️  Time taken: {:?}", start_t.elapsed());
}
