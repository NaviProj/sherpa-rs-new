/*
Transcribe wav file using SenseVoice

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example sense_voice motivation.wav
*/

use sherpa_rs::{
    get_default_provider, read_audio_file,
    sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer},
};

fn main() {
    let path = std::env::args().nth(1).expect("Missing file path argument");
    // Use CUDA provider by default, but allow override from command line
    let default_provider = if cfg!(feature = "cuda") { "cuda".to_string() } else { get_default_provider() };
    let provider = std::env::args().nth(2).unwrap_or(default_provider);
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    println!("Using provider: {}", provider);

    let config = SenseVoiceConfig {
        model: "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx".into(),
        tokens: "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt".into(),
        provider: Some(provider),
        debug: true, // Enable debug mode

        ..Default::default()
    };

    println!("Creating recognizer with config...");
    let mut recognizer: SenseVoiceRecognizer = match SenseVoiceRecognizer::new(config) {
        Ok(rec) => {
            println!("Recognizer created successfully!");
            rec
        },
        Err(e) => {
            eprintln!("Failed to create recognizer: {:?}", e);
            std::process::exit(1);
        }
    };

    println!("Starting transcription...");
    let start_t = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    let elapsed = start_t.elapsed();
    let audio_duration = samples.len() as f32 / sample_rate as f32;
    let rtf = elapsed.as_secs_f32() / audio_duration;

    println!("✅ Text: {}", result.text);
    println!("⏱️ Time taken for transcription: {:.3}s", elapsed.as_secs_f32());
    println!("🎵 Audio duration: {:.3}s", audio_duration);
    println!("⚡ Real Time Factor (RTF): {:.3}", rtf);
}
