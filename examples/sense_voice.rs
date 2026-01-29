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
    let provider = std::env::args().nth(2).unwrap_or(get_default_provider());
    let (samples, sample_rate) = read_audio_file(&path).unwrap();
    assert_eq!(sample_rate, 16000, "The sample rate must be 16000.");

    let config = SenseVoiceConfig {
        model: "/Users/trevorlink/Project/tiebao/NoteCapture/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx".into(),
        tokens: "/Users/trevorlink/Project/tiebao/NoteCapture/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt".into(),
        provider: Some(provider),

        ..Default::default()
    };

    let mut recognizer: SenseVoiceRecognizer = SenseVoiceRecognizer::new(config).unwrap();

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
