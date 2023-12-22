import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
from .CST_details.encoder import Encoder
from .CST_details.CST_encoder import CST_encoder
from .CST_details.CMT_Block import CMT_block
from .CST_details.layers import FC_layer

class CST_former(torch.nn.Module):
    """
    CST_former : Channel-Spectral-Temporal Transformer for SELD task
    """
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_unfold = params['ChAtten_ULE']
        self.cmt_block = params['CMT_block']
        self.encoder = Encoder(in_feat_shape, params)

        self.conv_block_freq_dim = int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.input_nb_ch = 7
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt'] * self.input_nb_ch if self.ch_attn_dca \
            else self.conv_block_freq_dim * params['nb_cnn2d_filt']

        ## Attention Layer===========================================================================================#
        if not self.cmt_block:
            self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        else:
            self.attention_stage = CMT_block(params, self.temp_embed_dim)


        if self.t_pooling_loc == 'end':
            if not params["f_pool_size"] == [1,1,1]:
                self.t_pooling = nn.MaxPool2d((5,1))
            else:
                self.t_pooling = nn.MaxPool2d((5,4))

        ## Fully Connected Layer ======================================================================================#
        self.fc_layer = FC_layer(out_shape, self.temp_embed_dim, params)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, video=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        B, M, T, F = x.size()

        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous()
        x = self.encoder(x) # OUT : [(b m) c t f] if ch_attn_dca else [b c t f]
        x = self.attention_stage(x)

        if self.t_pooling_loc == 'end':
            x = self.t_pooling(x)

        doa = self.fc_layer(x)

        return doa