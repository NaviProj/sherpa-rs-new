use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

use crate::utils::cstr_to_string;

pub struct StreamingParaformerRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
}

#[derive(Debug, Clone)]
pub struct StreamingParaformerConfig {
    pub encoder: String,
    pub decoder: String,
    pub tokens: String,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
    pub debug: bool,
    pub enable_endpoint: bool,
    pub rule1_min_trailing_silence: f32,
    pub rule2_min_trailing_silence: f32,
    pub rule3_min_utterance_length: f32,
}

impl Default for StreamingParaformerConfig {
    fn default() -> Self {
        Self {
            encoder: String::new(),
            decoder: String::new(),
            tokens: String::new(),
            debug: false,
            provider: None,
            num_threads: Some(1),
            enable_endpoint: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingRecognizerResult {
    pub text: String,
    pub count: i32,
}

pub struct OnlineStream {
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl StreamingParaformerRecognizer {
    pub fn new(config: StreamingParaformerConfig) -> Result<Self> {
        let debug = config.debug.into();
        let provider = config.provider.unwrap_or(get_default_provider());

        let provider_ptr = cstring_from_str(&provider);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let decoder_ptr = cstring_from_str(&config.decoder);
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let paraformer_config = sherpa_rs_sys::SherpaOnnxOnlineParaformerModelConfig {
            encoder: encoder_ptr.as_ptr(),
            decoder: decoder_ptr.as_ptr(),
        };

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                debug,
                num_threads: config.num_threads.unwrap_or(1),
                provider: provider_ptr.as_ptr(),
                tokens: tokens_ptr.as_ptr(),
                paraformer: paraformer_config,

                transducer: mem::zeroed::<_>(),
                zipformer2_ctc: mem::zeroed::<_>(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                bpe_vocab: mem::zeroed::<_>(),
                tokens_buf: mem::zeroed::<_>(),
                tokens_buf_size: 0,
                nemo_ctc: mem::zeroed::<_>(),
                t_one_ctc: mem::zeroed::<_>(),
            }
        };

        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig {
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                model_config,
                decoding_method: decoding_method_ptr.as_ptr(),
                max_active_paths: 0,
                enable_endpoint: config.enable_endpoint.into(),
                rule1_min_trailing_silence: config.rule1_min_trailing_silence,
                rule2_min_trailing_silence: config.rule2_min_trailing_silence,
                rule3_min_utterance_length: config.rule3_min_utterance_length,
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: 0.0,
                ctc_fst_decoder_config: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                rule_fars: mem::zeroed::<_>(),
                blank_penalty: 0.0,
                hotwords_buf: mem::zeroed::<_>(),
                hotwords_buf_size: 0,
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            bail!("Failed to create streaming Paraformer recognizer");
        }

        Ok(Self { recognizer })
    }

    pub fn create_stream(&self) -> Result<OnlineStream> {
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(self.recognizer) };
        if stream.is_null() {
            bail!("Failed to create online stream");
        }
        Ok(OnlineStream { stream })
    }

    pub fn is_ready(&self, stream: &OnlineStream) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, stream.stream) == 1
        }
    }

    pub fn decode(&self, stream: &OnlineStream) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, stream.stream);
        }
    }

    pub fn get_result(&self, stream: &OnlineStream) -> StreamingRecognizerResult {
        unsafe {
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, stream.stream);
            let raw_result = result_ptr.read();
            let text = cstr_to_string(raw_result.text);
            let count = raw_result.count;

            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);

            StreamingRecognizerResult { text, count }
        }
    }

    pub fn is_endpoint(&self, stream: &OnlineStream) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, stream.stream) == 1
        }
    }

    pub fn reset(&self, stream: &OnlineStream) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, stream.stream);
        }
    }
}

impl OnlineStream {
    pub fn accept_waveform(&self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    pub fn input_finished(&self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);
        }
    }
}

unsafe impl Send for StreamingParaformerRecognizer {}
unsafe impl Sync for StreamingParaformerRecognizer {}
unsafe impl Send for OnlineStream {}
unsafe impl Sync for OnlineStream {}

impl Drop for StreamingParaformerRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

impl Drop for OnlineStream {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
        }
    }
}
