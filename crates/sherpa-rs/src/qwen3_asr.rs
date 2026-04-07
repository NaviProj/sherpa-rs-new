use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{mem, ptr::null};

#[derive(Debug)]
pub struct Qwen3AsrRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type Qwen3AsrRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct Qwen3AsrConfig {
    pub conv_frontend: String,
    pub encoder: String,
    pub decoder: String,
    pub tokenizer: String,
    pub max_total_len: Option<i32>,
    pub max_new_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<i32>,
    pub hotwords: Option<String>,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for Qwen3AsrConfig {
    fn default() -> Self {
        Self {
            conv_frontend: String::new(),
            encoder: String::new(),
            decoder: String::new(),
            tokenizer: String::new(),
            max_total_len: None,
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            seed: None,
            hotwords: None,
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl Qwen3AsrRecognizer {
    pub fn new(config: Qwen3AsrConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let conv_frontend_ptr = cstring_from_str(&config.conv_frontend);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let decoder_ptr = cstring_from_str(&config.decoder);
        let tokenizer_ptr = cstring_from_str(&config.tokenizer);
        let hotwords_ptr = config
            .hotwords
            .as_deref()
            .map(cstring_from_str)
            .unwrap_or_else(|| cstring_from_str(""));
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let qwen3_asr_config = sherpa_rs_sys::SherpaOnnxOfflineQwen3ASRModelConfig {
            conv_frontend: conv_frontend_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            decoder: decoder_ptr.as_ptr(),
            tokenizer: tokenizer_ptr.as_ptr(),
            max_total_len: config.max_total_len.unwrap_or(0),
            max_new_tokens: config.max_new_tokens.unwrap_or(0),
            temperature: config.temperature.unwrap_or(0.0),
            top_p: config.top_p.unwrap_or(0.0),
            seed: config.seed.unwrap_or(0),
            hotwords: hotwords_ptr.as_ptr(),
        };

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads: config.num_threads.unwrap_or(1),
                provider: provider_ptr.as_ptr(),
                tokens: null(),
                qwen3_asr: qwen3_asr_config,

                // Null other model types
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: null(),
                transducer: mem::zeroed::<_>(),
                whisper: mem::zeroed::<_>(),
                sense_voice: mem::zeroed::<_>(),
                moonshine: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
                wenet_ctc: mem::zeroed::<_>(),
                omnilingual: mem::zeroed::<_>(),
                medasr: mem::zeroed::<_>(),
                funasr_nano: mem::zeroed::<_>(),
                fire_red_asr_ctc: mem::zeroed::<_>(),
                cohere_transcribe: mem::zeroed::<_>(),
            }
        };

        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: decoding_method_ptr.as_ptr(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                model_config,
                hotwords_file: null(),
                hotwords_score: 0.0,
                lm_config: mem::zeroed::<_>(),
                max_active_paths: 0,
                rule_fars: null(),
                rule_fsts: null(),
                blank_penalty: 0.0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create Qwen3-ASR recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(
        &mut self,
        sample_rate: u32,
        samples: &[f32],
    ) -> Qwen3AsrRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let result = Qwen3AsrRecognizerResult::new(&raw_result);

            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            result
        }
    }
}

unsafe impl Send for Qwen3AsrRecognizer {}
unsafe impl Sync for Qwen3AsrRecognizer {}

impl Drop for Qwen3AsrRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
