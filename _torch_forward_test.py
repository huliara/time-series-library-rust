# rustのモデルをテストする際、Rustから呼び出すコード
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from _args_mock import Args_mock
from models import (
    Autoformer,
    Transformer,
    TimesNet,
    Nonstationary_Transformer,
    DLinear,
    FEDformer,
    Informer,
    LightTS,
    Reformer,
    ETSformer,
    Pyraformer,
    PatchTST,
    MICN,
    Crossformer,
    FiLM,
    iTransformer,
    Koopa,
    TiDE,
    FreTS,
    TimeMixer,
    TSMixer,
    SegRNN,
    MambaSimple,
    TemporalFusionTransformer,
    SCINet,
    PAttn,
    TimeXer,
    WPMixer,
    MultiPatchFormer,
    KANAD,
    MSGNet,
    TimeFilter,
    Sundial,
    TimeMoE,
    Chronos,
    Moirai,
    TiRex,
    TimesFM,
    Chronos2,
)
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    my_Layernorm,
    moving_avg,
    series_decomp,
    series_decomp_multi,
    EncoderLayer as Autoformer_EncoderLayer,
    Encoder as Autoformer_Encoder,
    DecoderLayer as Autoformer_DecoderLayer,
    Decoder as Autoformer_Decoder,
)
from layers.Conv_Blocks import Inception_Block_V1, Inception_Block_V2
from layers.Crossformer_EncDec import (
    SegMerging,
    scale_block,
    Encoder as Crossformer_Encoder,
    DecoderLayer as Crossformer_DecoderLayer,
    Decoder as Crossformer_Decoder,
)
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse
from layers.Embed import (
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    PatchEmbedding,
)
from layers.ETSformer_EncDec import (
    ExponentialSmoothing,
    Feedforward,
    GrowthLayer,
    FourierLayer,
    LevelLayer,
    EncoderLayer as ETSformer_EncoderLayer,
    Encoder as ETSformer_Encoder,
    DampingLayer,
    DecoderLayer as ETSformer_DecoderLayer,
    Decoder as ETSformer_Decoder,
)
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MSGBlock import (
    Predict,
    Attention_Block,
    self_attention,
    FullAttention,
    GraphBlock,
    nconv,
    linear,
    mixprop,
    simpleVIT,
    MultiHeadAttention,
    FeedForward,
)
from layers.MultiWaveletCorrelation import (
    MultiWaveletTransform,
    MultiWaveletCross,
    FourierCrossAttentionW,
    sparseKernelFT1d,
    MWT_CZ1d,
)
from layers.Pyraformer_EncDec import (
    EncoderLayer as Pyraformer_EncoderLayer,
    Encoder as Pyraformer_Encoder,
    ConvLayer as Pyraformer_ConvLayer,
    Bottleneck_Construct,
    PositionwiseFeedForward,
)
from layers.SelfAttention_Family import (
    DSAttention,
    FullAttention,
    ProbAttention,
    AttentionLayer,
    ReformerLayer,
    TwoStageAttentionLayer,
)
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import (
    GCN,
    mask_moe,
    GraphLearner,
    GraphFilter,
    GraphBlock,
    TimeFilter_Backbone,
)
from layers.Transformer_EncDec import (
    ConvLayer as Transformer_ConvLayer,
    EncoderLayer as Transformer_EncoderLayer,
    Encoder as Transformer_Encoder,
    DecoderLayer as Transformer_DecoderLayer,
    Decoder as Transformer_Decoder,
)

layer_dict = {
    "AutoCorrelation": AutoCorrelation,
    "AutoCorrelationLayer": AutoCorrelationLayer,
    "my_Layernorm": my_Layernorm,
    "moving_avg": moving_avg,
    "series_decomp": series_decomp,
    "series_decomp_multi": series_decomp_multi,
    "Autoformer_EncoderLayer": Autoformer_EncoderLayer,
    "Autoformer_Encoder": Autoformer_Encoder,
    "Autoformer_DecoderLayer": Autoformer_DecoderLayer,
    "Autoformer_Decoder": Autoformer_Decoder,
    "Inception_Block_V1": Inception_Block_V1,
    "Inception_Block_V2": Inception_Block_V2,
    "SegMerging": SegMerging,
    "scale_block": scale_block,
    "Crossformer_Encoder": Crossformer_Encoder,
    "Crossformer_DecoderLayer": Crossformer_DecoderLayer,
    "Crossformer_Decoder": Crossformer_Decoder,
    "DWT1DForward": DWT1DForward,
    "DWT1DInverse": DWT1DInverse,
    "PositionalEmbedding": PositionalEmbedding,
    "TokenEmbedding": TokenEmbedding,
    "FixedEmbedding": FixedEmbedding,
    "TemporalEmbedding": TemporalEmbedding,
    "TimeFeatureEmbedding": TimeFeatureEmbedding,
    "DataEmbedding": DataEmbedding,
    "DataEmbedding_inverted": DataEmbedding_inverted,
    "DataEmbedding_wo_pos": DataEmbedding_wo_pos,
    "PatchEmbedding": PatchEmbedding,
    "ExponentialSmoothing": ExponentialSmoothing,
    "Feedforward": Feedforward,
    "GrowthLayer": GrowthLayer,
    "FourierLayer": FourierLayer,
    "LevelLayer": LevelLayer,
    "ETSformer_EncoderLayer": ETSformer_EncoderLayer,
    "ETSformer_Encoder": ETSformer_Encoder,
    "DampingLayer": DampingLayer,
    "ETSformer_DecoderLayer": ETSformer_DecoderLayer,
    "ETSformer_Decoder": ETSformer_Decoder,
    "FourierBlock": FourierBlock,
    "FourierCrossAttention": FourierCrossAttention,
    "Predict": Predict,
    "Attention_Block": Attention_Block,
    "self_attention": self_attention,
    "FullAttention": FullAttention,
    "GraphBlock": GraphBlock,
    "nconv": nconv,
    "linear": linear,
    "mixprop": mixprop,
    "simpleVIT": simpleVIT,
    "MultiHeadAttention": MultiHeadAttention,
    "FeedForward": FeedForward,
    "MultiWaveletTransform": MultiWaveletTransform,
    "MultiWaveletCross": MultiWaveletCross,
    "FourierCrossAttentionW": FourierCrossAttentionW,
    "sparseKernelFT1d": sparseKernelFT1d,
    "MWT_CZ1d": MWT_CZ1d,
    "Pyraformer_EncoderLayer": Pyraformer_EncoderLayer,
    "Pyraformer_Encoder": Pyraformer_Encoder,
    "Pyraformer_ConvLayer": Pyraformer_ConvLayer,
    "Bottleneck_Construct": Bottleneck_Construct,
    "PositionwiseFeedForward": PositionwiseFeedForward,
    "DSAttention": DSAttention,
    "ProbAttention": ProbAttention,
    "AttentionLayer": AttentionLayer,
    "ReformerLayer": ReformerLayer,
    "TwoStageAttentionLayer": TwoStageAttentionLayer,
    "Normalize": Normalize,
    "GCN": GCN,
    "mask_moe": mask_moe,
    "GraphLearner": GraphLearner,
    "GraphFilter": GraphFilter,
    "TimeFilter_Backbone": TimeFilter_Backbone,
    "Transformer_ConvLayer": Transformer_ConvLayer,
    "Transformer_EncoderLayer": Transformer_EncoderLayer,
    "Transformer_Encoder": Transformer_Encoder,
    "Transformer_DecoderLayer": Transformer_DecoderLayer,
    "Transformer_Decoder": Transformer_Decoder,
}

model_dict = {
    "TimesNet": TimesNet,
    "Autoformer": Autoformer,
    "Transformer": Transformer,
    "Nonstationary_Transformer": Nonstationary_Transformer,
    "DLinear": DLinear,
    "FEDformer": FEDformer,
    "Informer": Informer,
    "LightTS": LightTS,
    "Reformer": Reformer,
    "ETSformer": ETSformer,
    "PatchTST": PatchTST,
    "Pyraformer": Pyraformer,
    "MICN": MICN,
    "Crossformer": Crossformer,
    "FiLM": FiLM,
    "iTransformer": iTransformer,
    "Koopa": Koopa,
    "TiDE": TiDE,
    "FreTS": FreTS,
    "MambaSimple": MambaSimple,
    "TimeMixer": TimeMixer,
    "TSMixer": TSMixer,
    "SegRNN": SegRNN,
    "TemporalFusionTransformer": TemporalFusionTransformer,
    "SCINet": SCINet,
    "PAttn": PAttn,
    "TimeXer": TimeXer,
    "WPMixer": WPMixer,
    "MultiPatchFormer": MultiPatchFormer,
    "KANAD": KANAD,
    "MSGNet": MSGNet,
    "TimeFilter": TimeFilter,
    "Sundial": Sundial,
    "TimeMoE": TimeMoE,
    "Chronos": Chronos,
    "Moirai": Moirai,
    "TiRex": TiRex,
    "TimesFM": TimesFM,
    "Chronos2": Chronos2,
}


def torch_forward_test(name):
    args = Args_mock()
    exp = Exp_Long_Term_Forecast(args)
    device = exp.device
    module: nn.Module = model_dict[name].Model(args).float()
    module.to(device)
    module.eval()
    if name == "DLinear":
        # Initialize DLinear weights and biases for testing
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.01)
    elif name == "PatchTST":
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, 0.01)
            elif "bias" in name and "norm" in name:
                nn.init.constant_(param, 0.00)
            elif "bias" in name:
                nn.init.constant_(param, 0.01)
    else:
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, 0.01)
            elif "bias" in name:
                nn.init.constant_(param, 0.01)
    _, data_loader = data_provider(args, flag="test")
    all_outputs = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )
            # encoder - decoder
            if args.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = module(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = module(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]

            all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    print("all_outputs shape:", all_outputs.shape)
    return all_outputs


if __name__ == "__main__":
    output = torch_forward_test("PatchTST")
    print("output shape:", output.shape)
