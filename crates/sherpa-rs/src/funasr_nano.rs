use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

#[derive(Debug)]
pub struct FunASRNanoRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

pub type FunASRNanoRecognizerResult = super::OfflineRecognizerResult;

#[derive(Debug, Clone)]
pub struct FunASRNanoConfig {
    pub encoder_adaptor: String,
    pub llm: String,
    pub embedding: String,
    pub tokenizer: String,
    pub language: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
}

impl FunASRNanoRecognizer {
    pub fn new(config: FunASRNanoConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());
        let provider_ptr = cstring_from_str(&provider);
        let num_threads = config.num_threads.unwrap_or(1);

        let encoder_adaptor_ptr = cstring_from_str(&config.encoder_adaptor);
        let llm_ptr = cstring_from_str(&config.llm);
        let embedding_ptr = cstring_from_str(&config.embedding);
        let tokenizer_ptr = cstring_from_str(&config.tokenizer);
        let language_ptr = cstring_from_str(&config.language);

        let funasr_nano_config = sherpa_rs_sys::SherpaOnnxOfflineFunASRNanoModelConfig {
            encoder_adaptor: encoder_adaptor_ptr.as_ptr(),
            llm: llm_ptr.as_ptr(),
            embedding: embedding_ptr.as_ptr(),
            tokenizer: tokenizer_ptr.as_ptr(),
            system_prompt: std::ptr::null(),
            user_prompt: std::ptr::null(),
            max_new_tokens: 0,
            temperature: 0.0,
            top_p: 0.0,
            seed: 0,
            language: language_ptr.as_ptr(),
            itn: 0,
            hotwords: std::ptr::null(),
        };

        // Empty tokens string — FunASR-nano uses its own tokenizer
        let tokens_ptr = cstring_from_str("");

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                tokens: tokens_ptr.as_ptr(),
                provider: provider_ptr.as_ptr(),
                num_threads,
                debug,
                funasr_nano: funasr_nano_config,
                bpe_vocab: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                paraformer: mem::zeroed::<_>(),
                tdnn: mem::zeroed::<_>(),
                telespeech_ctc: mem::zeroed::<_>(),
                fire_red_asr: mem::zeroed::<_>(),
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
            }
        };

        let config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: mem::zeroed::<_>(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: 0.0,
                lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                    model: mem::zeroed::<_>(),
                    scale: 0.0,
                },
                max_active_paths: 0,
                model_config,
                rule_fars: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                blank_penalty: 0.0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };
        if recognizer.is_null() {
            bail!("Failed to create FunASR-nano recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn transcribe(
        &mut self,
        sample_rate: u32,
        samples: &[f32],
    ) -> FunASRNanoRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let result = FunASRNanoRecognizerResult::new(&raw_result);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            result
        }
    }
}

unsafe impl Send for FunASRNanoRecognizer {}
unsafe impl Sync for FunASRNanoRecognizer {}

impl Drop for FunASRNanoRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
