use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::{mem, ptr::null};

#[derive(Debug)]
pub struct FireRedAsrRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type FireRedAsrRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct FireRedAsrConfig {
    pub encoder: String,
    pub decoder: String,
    pub tokens: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl Default for FireRedAsrConfig {
    fn default() -> Self {
        Self {
            encoder: String::new(),
            decoder: String::new(),
            tokens: String::new(),
            debug: false,
            provider: None,
            num_threads: Some(1),
        }
    }
}

impl FireRedAsrRecognizer {
    pub fn new(config: FireRedAsrConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let decoder_ptr = cstring_from_str(&config.decoder);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let fire_red_asr_config = sherpa_rs_sys::SherpaOnnxOfflineFireRedAsrModelConfig {
            encoder: encoder_ptr.as_ptr(),
            decoder: decoder_ptr.as_ptr(),
        };

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug,
                num_threads: config.num_threads.unwrap_or(1),
                provider: provider_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),
                fire_red_asr: fire_red_asr_config,

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
                dolphin: mem::zeroed::<_>(),
                zipformer_ctc: mem::zeroed::<_>(),
                canary: mem::zeroed::<_>(),
                wenet_ctc: mem::zeroed::<_>(),
                omnilingual: mem::zeroed::<_>(),
                medasr: mem::zeroed::<_>(),
                funasr_nano: mem::zeroed::<_>(),
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
            bail!("Failed to create FireRedASR recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(
        &mut self,
        sample_rate: u32,
        samples: &[f32],
    ) -> FireRedAsrRecognizerResult {
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
            let result = FireRedAsrRecognizerResult::new(&raw_result);

            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            result
        }
    }
}

unsafe impl Send for FireRedAsrRecognizer {}
unsafe impl Sync for FireRedAsrRecognizer {}

impl Drop for FireRedAsrRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
