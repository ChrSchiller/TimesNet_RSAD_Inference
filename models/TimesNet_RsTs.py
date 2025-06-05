import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1 # , Inception_Block_V2

### this class avoids an error that occurs because the GELU activation function does not accept two arguments
### but the Inception_Block expects two arguments for the activation function
### so the GELUwrapper essentially just drops the doy argument
class GELUWrapper(nn.Module):
    def __init__(self):
        super(GELUWrapper, self).__init__()
        self.gelu = nn.GELU()
    
    def forward(self, x, _):
        return self.gelu(x)

def apply_sine_embedding(time_tensor, max_time=366):
    """
    This function applies the sine/cosine embedding to the time tensor.
    Assumes the time tensor is in days and the maximum time value is max_time (e.g., 365 for days of the year).
    """
    time_tensor = time_tensor / max_time  # Normalize to range [0, 1]

    # Apply sine and cosine transformations
    sin_embedding = torch.sin(2 * torch.pi * time_tensor) # range [0, 2 pi] -> [-1, 1]
    # cos_embedding = torch.cos(2 * torch.pi * time_tensor)

    # # Concatenate sine and cosine embeddings along the last dimension
    # time_embedding = torch.cat([sin_embedding, cos_embedding], dim=-1)

    # return time_embedding
    return sin_embedding

### note that we changed k to a larger model, because we have a 10-variate complex time series
### top_k parameter set in the config file
def FFT_for_Period(x, k=2):  # default: k=2, but this is a hyperparameter
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = 0
        self.k = configs.top_k
        # parameter-efficient design
        ### note that we changed the default of Inception_Block_V1 to Inception_Block_V2 (changed back already, 2025-01-03)
        ### that means instead of symmetrical convolutions, we have asymmetrical convolutions
        ### which either operate ONLY across the token embedding dimension, 
        ### or ONLY across the time dimension
        ### Inception_Block_V1 always operates across both dimensions with its kernels
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            GELUWrapper(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        
        
    def forward(self, x, doy):
        B, T, N = x.size()
        # N is d_model
        # period_list, period_weight = FFT_for_Period(x, self.k)
        ### we dropped Fourier transforms here, because it does not make sense in irregular time series

        res = []
        ### we might want to use a period_list again, 
        ### since it might make sense to take the full sequence and the yearly sequences (two periods)
        # for i in range(self.k):
        # period = period_list[i]
        ### we re-define period, which is always the length of the whole sequence divided by 4
        ### this is because we have a 4 year time series to run the convolutions over
        period = x.shape[1] // 4 ## seq_len is 200, so this will be 50

        # padding
        if (self.seq_len + self.pred_len) % period != 0:
            length = (
                                ((self.seq_len + self.pred_len) // period) + 1) * period
            padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = (self.seq_len + self.pred_len)
            out = x
        
        # reshape
        out = out.reshape(B, length // period, period,
                            N).permute(0, 3, 1, 2).contiguous()
        # shape is [B, d_model, seq_len // period, period]
        doy = doy.reshape(B, length // period, period)  # Reshape DOY for convolution
        # shape is [B, seq_len // period, period]

        # 2D conv: from 1d Variation to 2d Variation
        # out = self.conv(out, doy) # same shape before and after the operation
        ### apply each layer in the sequential container individually 
        ### (otherwise error occurs: does not accept two arguments)
        for layer in self.conv:
            out = layer(out, doy)
        
        # reshape back
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        ### results in shape [batch_size, seq_len + pred_len, d_model]

        ### the following two lines can be simplified (or even dropped?) because they stem from top_k > 1 from Fourier transform
        ### but in the latest code we only have 1 period, so top_k == 1
        res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1).squeeze(-1)
        ### the squeeze operation on the last dimension stems from the fact that we only have top_k == 1
        ### because it is the number of list entries from the list "res"
        ### in the original code, the following commented out lines are used for top_k > 1
        ### which contain an aggregation to remove this last dimension
        ### we don't need that, but we need to squeeze instead 
        ### (or use the aggregation step as well, 
        ### but it is not necessary for top_k == 1)

        ### hopefully, it is enough to just drop the adaptive aggregation step, 
        ### which is only necessary in case of top_k > 1
        ### we also don't have period_weight anymore
        # # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res # [B, seq_len, d_model]


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # self.label_len = configs.label_len
        self.pred_len = 0
        ### this sequential nature of the TimesBlock stack means that the consecutive TimesBlock gets the 
        ### output of the previous one, and only the first TimesBlock gets the real input data as input
        ### since there is no pooling for higher-level feature extraction, this is more akin to 
        ### a refinement of the features learned from the previous layers/blocks
        ### nevertheless: Each block essentially has access to the output from the previous block, 
        ### allowing it to learn higher-level abstractions over time. So, the model does not explicitly create a "hierarchical" abstraction 
        ### in the sense of pooling layers in CNNs; 
        ### it refines the same sequence progressively across layers
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, len(configs.indices_bands))
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    # def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(
    #         torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #     x_enc /= stdev

    #     # embedding
    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
    #     enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
    #         0, 2, 1)  # align temporal dimension
    #     # TimesNet
    #     for i in range(self.layer):
    #         enc_out = self.layer_norm(self.model[i](enc_out))
    #     # project back
    #     dec_out = self.projection(enc_out)

    #     # De-Normalization from Non-stationary Transformer
    #     dec_out = dec_out * \
    #               (stdev[:, 0, :].unsqueeze(1).repeat(
    #                   1, self.pred_len + self.seq_len, 1))
    #     dec_out = dec_out + \
    #               (means[:, 0, :].unsqueeze(1).repeat(
    #                   1, self.pred_len + self.seq_len, 1))
    #     return dec_out

    # def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
    #     # Normalization from Non-stationary Transformer
    #     means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
    #     means = means.unsqueeze(1).detach()
    #     x_enc = x_enc - means
    #     x_enc = x_enc.masked_fill(mask == 0, 0)
    #     stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
    #                        torch.sum(mask == 1, dim=1) + 1e-5)
    #     stdev = stdev.unsqueeze(1).detach()
    #     x_enc /= stdev

    #     # embedding
    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
    #     # TimesNet
    #     for i in range(self.layer):
    #         enc_out = self.layer_norm(self.model[i](enc_out))
    #     # project back
    #     dec_out = self.projection(enc_out)

    #     # De-Normalization from Non-stationary Transformer
    #     dec_out = dec_out * \
    #               (stdev[:, 0, :].unsqueeze(1).repeat(
    #                   1, self.pred_len + self.seq_len, 1))
    #     dec_out = dec_out + \
    #               (means[:, 0, :].unsqueeze(1).repeat(
    #                   1, self.pred_len + self.seq_len, 1))
    #     return dec_out

    ### we added x_mark_enc here, which constitutes the day-of-the-year information
    def anomaly_detection(self, x_enc, x_mark_enc): 
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        # Create a mask that is True for the non-padded part and False for the padded part
        
        ### Normalization from Non-stationary Transformer
        ### we need to exclude the padding values (end padding with zeros)
        ### from normalization and de-normalization
        mask = (x_enc != 0)
        # means = (x_enc * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        # x_enc -= means
        # variance = (mask * (x_enc ** 2)).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        # stdev = torch.sqrt(variance + 1e-5)

        # stdev = torch.sqrt((mask * (x_enc ** 2)).sum(1, keepdim=True) / mask.sum(1, keepdim=True) + 1e-5)
        # stdev = torch.sqrt((mask * (x_enc ** 2)).sum(1, keepdim=True) / mask.sum(1, keepdim=True) + 1e-5)

        # # Normalize the non-padded part
        # x_enc = mask * (x_enc / stdev)

        ### note that there is no normalization of x_mark/x_mark_enc (= DOY) in neither of the temporal embeddings
        ### while this makes sense for nn.Embedding, we need to think about it for sin cos embeddings

        # embedding
        ### if no x_mark_enc is passed, the model will use the default value of None
        ### x_mark is only passed in forecast, imputation and classification by default
        ### but not in anomaly detection
        ### passing x_mark leads to explicit Time Embedding
        ### this is what we want
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] 
        ### enc_out.shape does not change despite the explicit time information, but the embeddings are different
        # TimesNet
        ### this should work as in the default, because the shape remains the same
        for i in range(self.layer):
            ### apply the mask to the input of the model
            # select the first "column" of mask and reshape it for broadcasting
            mask_broadcast = mask[:, :, 0].unsqueeze(-1)
            # broadcast mask_broadcast to match the shape of enc_out and multiply
            enc_out = enc_out * mask_broadcast

            ### normalize the x_mark_enc time tensor and mask it to make sure 0 values are not considered
            x_mark_enc = apply_sine_embedding(x_mark_enc, max_time=366)
            ### mask x_mark_enc to make sure 0 values are not considered
            x_mark_enc = x_mark_enc * mask[:, :, 0]

            enc_out = self.layer_norm(self.model[i](enc_out, x_mark_enc))
        # project back
        dec_out = self.projection(enc_out) # shape [B, T, C] again

        # # De-Normalization from Non-stationary Transformer
        # ### I think pred_len is not relevant for anomaly_detection, only for forecasting!?
        # ### multiplying by mask probably not necessary, because we don't have padding in the output
        # ### we apply it anyway to be consistent with the other functions
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)) * mask
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)) * mask
        ### we skip the de-normalization for anomaly detection as we did not normalize beforehand
        dec_out = dec_out * mask
        return dec_out # reconstructed time series [B, T, C]

    # def classification(self, x_enc, x_mark_enc):
    #     # embedding
    #     enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
    #     # TimesNet
    #     for i in range(self.layer):
    #         enc_out = self.layer_norm(self.model[i](enc_out))

    #     # Output
    #     # the output transformer encoder/decoder embeddings don't include non-linearity
    #     output = self.act(enc_out)
    #     output = self.dropout(output)
    #     # zero-out padding embeddings
    #     output = output * x_mark_enc.unsqueeze(-1)
    #     # (batch_size, seq_length * d_model)
    #     output = output.reshape(output.shape[0], -1)
    #     output = self.projection(output)  # (batch_size, num_classes)
    #     return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        ### this is the "real" forward function that passes the input data for the model
        ### it is what is being executed when executing self.model(...)
        ### so we need to add x_mark here! and of course also in the def anomaly_detection itself
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
