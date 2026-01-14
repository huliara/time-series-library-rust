use crate::args::{Args, TaskName};
use crate::layers::embed::DataEmbedding;
use crate::layers::self_attention_family::{AttentionLayer, FullAttention};
use crate::layers::transformer_enc_dec::{Decoder, DecoderLayer, Encoder, EncoderLayer};
use burn::module::Module;
use burn::nn::{LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    enc_embedding: DataEmbedding<B>,
    encoder: Encoder<B>,
    dec_embedding: Option<DataEmbedding<B>>,
    decoder: Option<Decoder<B>>,
    projection: Option<Linear<B>>,
    task_name: String,
}

impl<B: Backend> Transformer<B> {
    pub fn new(args: &Args, device: &B::Device) -> Self {
        let enc_embedding = DataEmbedding::new(
            args.enc_in,
            args.d_model,
            args.embed.clone(),
            args.freq.clone(),
            args.dropout,
            device,
        );

        let mut encoder_layers = Vec::new();
        for _ in 0..args.e_layers {
            let attention = AttentionLayer::new(
                FullAttention::<B>::new(false, args.factor, None, args.dropout, false),
                args.d_model,
                args.n_heads,
                None,
                None,
                device,
            );
            encoder_layers.push(EncoderLayer::new(
                attention,
                args.d_model,
                Some(args.d_ff),
                args.dropout,
                args.activation.clone(),
                device,
            ));
        }
        let encoder = Encoder::new(
            encoder_layers,
            Some(LayerNormConfig::new(args.d_model).init(device)),
        );

        let mut dec_embedding = None;
        let mut decoder = None;
        let mut projection = None;

        if args.task_name == TaskName::LongTermForecast
            || args.task_name == TaskName::ShortTermForecast
        {
            dec_embedding = Some(DataEmbedding::new(
                args.dec_in,
                args.d_model,
                args.embed.clone(),
                args.freq.clone(),
                args.dropout,
                device,
            ));

            let mut decoder_layers = Vec::new();
            for _ in 0..args.d_layers {
                let self_attention = AttentionLayer::new(
                    FullAttention::<B>::new(true, args.factor, None, args.dropout, false),
                    args.d_model,
                    args.n_heads,
                    None,
                    None,
                    device,
                );
                let cross_attention = AttentionLayer::new(
                    FullAttention::<B>::new(false, args.factor, None, args.dropout, false),
                    args.d_model,
                    args.n_heads,
                    None,
                    None,
                    device,
                );
                decoder_layers.push(DecoderLayer::new(
                    self_attention,
                    cross_attention,
                    args.d_model,
                    Some(args.d_ff),
                    args.dropout,
                    args.activation.clone(),
                    device,
                ));
            }
            decoder = Some(Decoder::new(
                decoder_layers,
                Some(LayerNormConfig::new(args.d_model).init(device)),
                LinearConfig::new(args.d_model, args.c_out).init(device),
            ));
        } else if args.task_name == TaskName::Imputation
            || args.task_name == TaskName::AnomalyDetection
        {
            projection = Some(LinearConfig::new(args.d_model, args.c_out).init(device));
        }

        Self {
            enc_embedding,
            encoder,
            dec_embedding,
            decoder,
            projection,
            task_name: format!("{:?}", args.task_name),
        }
    }

    pub fn forward(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Option<Tensor<B, 3>>,
        x_dec: Tensor<B, 3>,
        x_mark_dec: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        if self.task_name == "LongTermForecast" || self.task_name == "ShortTermForecast" {
            self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        } else if self.task_name == "Imputation" {
            self.imputation(x_enc, x_mark_enc)
        } else if self.task_name == "AnomalyDetection" {
            self.anomaly_detection(x_enc)
        } else {
            panic!("Task not implemented: {}", self.task_name);
        }
    }

    fn forecast(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Option<Tensor<B, 3>>,
        x_dec: Tensor<B, 3>,
        x_mark_dec: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let enc_out = self.enc_embedding.forward(x_enc, x_mark_enc);
        let (enc_out, _) = self.encoder.forward(enc_out, None);

        let dec_embedding = self.dec_embedding.as_ref().unwrap();
        let decoder = self.decoder.as_ref().unwrap();

        let dec_out = dec_embedding.forward(x_dec, x_mark_dec);
        decoder.forward(dec_out, enc_out, None, None)
    }

    fn imputation(&self, x_enc: Tensor<B, 3>, x_mark_enc: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let enc_out = self.enc_embedding.forward(x_enc, x_mark_enc);
        let (enc_out, _) = self.encoder.forward(enc_out, None);

        self.projection.as_ref().unwrap().forward(enc_out)
    }

    fn anomaly_detection(&self, x_enc: Tensor<B, 3>) -> Tensor<B, 3> {
        let enc_out = self.enc_embedding.forward(x_enc, None);
        let (enc_out, _) = self.encoder.forward(enc_out, None);

        self.projection.as_ref().unwrap().forward(enc_out)
    }
}
