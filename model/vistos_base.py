'''
VisTOS base module, based on the Presto-architecture (Tseng et al., 2024), which 
offers basic objects used in both VisTOS model types, the attention-based spatial 
encoding model and the convolution-based spatial encoding model. It comprises the
attention modules (Block objects) and auxiliary embedding classes. 
'''
import math
from typing import Tuple, Union, cast

import numpy as np
import torch
from einops import repeat
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

import params


class Attention(nn.Module):
    '''
    Multi-head attention module. Projects input to query, key, and value, 
    splits each into given number of attention heads and performs 
    scaled_dot_product attention if it is available. If not available, 
    uses a fallback option.

    Credit: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    '''

    # flag: F.scaled_dot_product_attention available
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=params.NUM_HEADS,
        qkv_bias=params.QKV_BIAS,
        qk_norm=False,
        attn_drop=params.DROPOUT,
        proj_drop=params.DROPOUT,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # embedding dimension must be divisible by number of heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads" 
        # number of attention-heads
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads # dimensions per head
        # scale factor for stabilization
        self.scale = self.head_dim**-0.5 
        # scaled_dot_product_attention available or not
        self.fast_attn = hasattr(F, "scaled_dot_product_attention")  
        # project input to query, key, value (3 x input)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # norms for query and key
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # dropout layer
        self.attn_drop = nn.Dropout(attn_drop)
        # fuse all heads outputs
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        # mini-batch size B, sequence length N, number of channels C
        B, N, C = x.shape
        # calculate query, key and value -> reshape to (3, B, num_head, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # create separate query, key and value (each: (B, num_head, N, head_dim))
        q, k, v = qkv.unbind(0)
        # apply layer norm
        q, k = self.q_norm(q), self.k_norm(k)

        # if F.scaled_dot_product_attention is available
        if self.fast_attn:
            # GPU bug: B*num_heads expected: (B, num_heads, N, head_dim) -> (B*num_heads, N, head_dim)
            B_head=B*self.num_heads
            q=q.reshape(B_head,N,self.head_dim)
            k=k.reshape(B_head,N,self.head_dim)
            v=v.reshape(B_head,N,self.head_dim)
            # if attention mask is available: reshape mask to (B,num_heads,N,N)
            if attn_mask is not None:               
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
                # GPU bug: B*num_heads expected (B, num_heads, N, N) -> (B*num_heads, N, N)
                attn_mask=attn_mask.reshape(B_head,N,N)
            # attention
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # False indicates that the element should take part in attention
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
            # reshape (B*num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
            x=x.reshape(B, self.num_heads, N, self.head_dim)
        else:
            # fallback option -> no attention mask implemented
            if attn_mask is not None:
                raise NotImplementedError
            # apply scaling to query
            q = q * self.scale
            # attention
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        # project back to model hidden dim and dropout
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    '''
    MLP as used in Vision Transformer, MLP-Mixer and related networks. Has 
    a GELU-nonlinearity layer, two linear layers and two dropout layers.
    Comes after multi-head attention, residual and layer norm.
    '''

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=params.DROPOUT,
    ):
        super().__init__()
        # output dimension: use input dimension, if nothing is passed
        out_features = out_features or in_features
        # hidden dimension: use input dimension, if nothing is passed
        hidden_features = hidden_features or in_features

        # linear layer 1
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        # GELU activation function
        self.act = act_layer()
        # dropout layer 1
        self.drop1 = nn.Dropout(drop)
        # linear layer 2
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        # dropout layer 2
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LayerScale(nn.Module):
    '''
    Gamma is a learnable scale layer applied to the residual connection, to define
    residual contribution. Is initialized weakly, during training, gamma learns 
    how much residual to contribute.
    '''

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        # learnable factor, applied per component of embedding dimension
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    '''
    Comprises an attention module, which consists of a layer norm, followed by a 
    multi-head attention module, a multi-layer perceptron, and again a layer norm. 
    Additionally, there are two residual connections with each an optional layer 
    scale.
    '''

    def __init__(
        self,
        dim,
        num_heads=params.NUM_HEADS,
        mlp_ratio=params.MLP_RATIO,
        qkv_bias=params.QKV_BIAS,
        qk_norm=False,
        drop=params.DROPOUT,
        attn_drop=params.DROPOUT,
        init_values=params.LAYER_SCALE,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # layer norm 1
        self.norm1 = norm_layer(dim)
        # multi-head attention module 
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        # layer scale 1
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # layer norm 2
        self.norm2 = norm_layer(dim)
        # multi-layer perceptron
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        # layer scale 2
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    '''
    Sinusoidal position encoding table for temporal positions
    (meaning month time steps): int or list of integer, if 
    int range(positions).
    '''

    # if only single int is passed, create a list of positions
    if isinstance(positions, int):
        positions = list(range(positions))

    # calculates angle for certain position and certain embedding dimension 
    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    # calculates angle for certain position over all dimensions
    def get_posi_angle_vec(position):
        # calls calculation of angles for each component of embedding dimension
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    # tabel with angle vector for every position (calls nested calculation of angles for certain position over all positions)
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    # apply sine on every even embedding position
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # apply cosine on every odd embedding position
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).to(params.DEVICE)

def get_month_encoding_table(d_hid):
    '''
    Sinusoidal month encoding table, for 12 months indexed from 0-11. Encodes
    each month with a cyclic sinusoidal encoding.
    '''

    # embedding dimensions must be divisible by 2
    assert d_hid % 2 == 0
    # creates 13 angles
    angles = np.arange(0, 13) / (12 / (2 * np.pi))

    # for half of embedding dimension stack sine applied to angles -> (13, d_hid/2)
    sin_table = np.sin(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    # for half of embedding dimension stack cosine applied to angles -> (13, d_hid/2)
    cos_table = np.cos(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    # concatenate last dimension, remove last entry in first dimension (repetitive) -> (12, d_hid)
    month_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    # return as tensor 
    if torch.cuda.is_available():
        return torch.FloatTensor(month_table).cuda()
    else:
        return torch.FloatTensor(month_table)

def month_to_tensor(month: Union[torch.Tensor, int], batch_size: int, seq_len: int):
    '''
    Creates a tensor of 12 contiguous months in length seq_len for passed month index 
    as integer. The result has the dimension (seq_len) For a tensor of month indices, 
    a sequence of contiguous month integer in length seq_len is created for each passed 
    month index, the result has the dimensions (passed months, seq_len); also returned 
    as tensor.
    '''

    # if single int passed, must encode a month as int -> values [0,11]
    if isinstance(month, int):
        assert cast(int, month) < 12
    # also tensor values must be in range [0,11]
    else:
        assert max(cast(torch.Tensor, month.flatten())) < 12

    # if single int is passed, create sequence of months as tensor
    if isinstance(month, int):
        # >>> torch.fmod(torch.tensor([9., 10, 11, 12, 13, 14]), 12)
        # tensor([ 9., 10., 11.,  0.,  1.,  2.])
        month = (
            torch.fmod(torch.arange(month, month + seq_len, dtype=torch.long), 12)
            .expand(batch_size, seq_len)
            .to(params.DEVICE)
        )
    # if several months are passed as tensor, create a matrix with a sequence of months starting at 
    # each passed month
    elif len(month.shape) == 1:
        month = torch.stack(
            [torch.fmod(torch.arange(m, m + seq_len, dtype=torch.long), 12) for m in month]
        ).to(params.DEVICE)
    return month

class EncoderBase(nn.Module):
    '''
    Encoder base module. Encoder classes of attention-based spatial encoding model and 
    convolution-based spatial encoding model inherit from this class basic functionality.
    The forward() function is an abstract method and implemented in the respective subclass.
    Offers building blocks for pixel-wise positional embedding of the model input 
    (multi-channel pixel time series in sequential order), the first attention block, which 
    performs a temporal self-attention encoding.  For fine-tuning, the time dimension can be 
    aggregated via a set of hierarchical convolutions and a GELU-activation function.
    '''

    def __init__(
        self,
        embedding_size: int = params.EMBEDDING_SIZE,
        channel_embed_ratio: float = params.CHANNEL_EMBED_RATIO,
        month_embed_ratio: float = params.MONTHS_EMBED_RATIO,
        depth: int =params.DEPTH,
        mlp_ratio: int =params.MLP_RATIO,
        num_heads: int =params.NUM_HEADS,
        max_sequence_length:int =params.MAX_SEQ_LEN,
        vis_field_size:int=params.VIS_FIELDS[0],
        mode: str='pretrain', 
    ):
        super().__init__()
        # pretraining or fine-tuning: ('pretrain', 'finetune')
        self.mode=mode
        # side length of the square visual field
        self.vis_field_size=vis_field_size
        # padding size for unfolded input 
        self.pad=vis_field_size//2
        # channel groups of the input
        self.band_groups = params.CHANNEL_GROUPS
        # embedding dimension of the encoder
        self.embedding_size = embedding_size
        # sequence length is variable due to compression -> pooling to fixed length
        self.output_seq_len= 80
        # reduce dynamic time dimension to fixed length
        self.adapt_pool=nn.AdaptiveAvgPool1d(output_size=self.output_seq_len)
        # merge time dimension
        merge_time_kernel=self.output_seq_len//2
        # 1. time convolution-> reduce to t_new/2
        self.merge_time_1= nn.Conv1d(embedding_size, embedding_size, kernel_size=2, stride=2)
        # GELU for aggregating time dimension
        self.merge_time_non_linear=nn.GELU()
        # 2. time convolution -> reduce to 1
        self.merge_time_2= nn.Conv1d(embedding_size, embedding_size, kernel_size=merge_time_kernel, stride=1)
        # set mode with set_mode()-> defines img_width
        self.img_width = params.IMG_WIDTH 

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: data['id'] for group_name,data in self.band_groups.items()
        }
        self.band_group_to_idx['DW'] = max(self.band_group_to_idx.values()) + 1

        # token embedding
        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(data['idx']), embedding_size)
                for group_name, data in params.CHANNEL_GROUPS.items()
            }
        )
        self.dw_embed = nn.Embedding(
            num_embeddings=params.DW_CLASSES + 1, embedding_dim=embedding_size
        )
        self.latlon_embed = nn.Linear(3, embedding_size)

        # temporal encoding blocks
        self.block_1 = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        

        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        # length of position embedding in embedding dimension
        pos_embedding_size = int(embedding_size * (1 - (channel_embed_ratio + month_embed_ratio)))
        # length of channel embedding in embedding dimension
        channel_embedding_size = int(embedding_size * channel_embed_ratio)
        # length of month embedding in embedding dimension
        month_embedding_size = int(embedding_size * month_embed_ratio)
        # fixed sinusoidal position embedding 
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )
        # month table made from input month sequence
        month_tab = get_month_encoding_table(month_embedding_size)
        # embed month table 
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        # embed channel group 
        self.channel_embed = nn.Embedding(
            num_embeddings=len(params.CHANNEL_GROUPS) + 1, embedding_dim=channel_embedding_size
        )

    def initialize_weights(self):
        '''
        Initializes embedding weights and model weights via self.apply(). 
        '''

        # initializes position sinusoidal encoding 
        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        # copy values into position embedding 
        self.pos_embed.data.copy_(pos_embed)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        Initializes the encoder weights via self.apply() with Xavier-uniform
        for linear layers, bias of linear layer as 0.0. Initializes layer norm
        with 1.0 and bias of layer norm with 0.0.
        '''

        # linear layer
        if isinstance(m, nn.Linear):
            # use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # layer norm
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_img_width(self,img_width):
        '''
        Set width of input image (CDDS and PASTIS-R) for spatial
        encoding. 
        '''

        self.img_width=img_width

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        '''
        Converts latitude/longitude coordinates into cartesian 3D unit vectors
        (x, y, z):
        
        x= cos(latitude)*cos(longitude)
        y= cos(latitude)*sin(longitude)
        z= sin(latitude)
        '''
        with torch.no_grad():
            # an embedding is calculated for all timesteps. This is then expanded
            # for each timestep in the sequence
            latlon_radians = latlons * math.pi / 180
            lats, lons = latlon_radians[:, 0], latlon_radians[:, 1]
            x = torch.cos(lats) * torch.cos(lons)
            y = torch.cos(lats) * torch.sin(lons)
            z = torch.sin(lats)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def mask_tokens(x, mask):
        '''
        Compression method: Sorts per batch (b=batch size, t=time steps, e=embedding dimension) all
        non-masked time steps to the front. Crops the entire batch in the time dimension (2nd dimension)
        to the maximum length of unmasked time steps in the batch. Saves the original indices in the 
        return variable of the same name, applies the compression to the input and the mask. 
        '''

        # convert integer mask to boolean
        mask = mask.bool()
        # https://stackoverflow.com/a/68621610/2332296
        # move all non-masked values to the front of their rows
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # set masked values to 0 (not really necessary since they are ignored anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # crop to the length of the longest sequence in the batch (time steps)
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask

    def forward(
        self,
        input_dict: dict,
        eval_task: bool = True,
    ):
        '''
        Conducts forward pass of the encoder: has to be implemented in subclasses.
        '''

        raise NotImplementedError

class DecoderBase(nn.Module):
    '''
    Base class for the decoder module used in the VisTOS pretraining autoencoder
    architecture. The decoder reconstructs from the encoder representation the 
    Earth observation data and the Dynamic World data. 
    '''

    def __init__(
        self,
        channel_embeddings: nn.Embedding,
        encoder_embed_dim=params.EMBEDDING_SIZE,
        decoder_embed_dim=params.EMBEDDING_SIZE,
        decoder_depth=params.DEPTH,
        decoder_num_heads=params.NUM_HEADS,
        mlp_ratio=params.DECODER_MLP_RATIO,
        max_sequence_length=params.MAX_SEQ_LEN,
    ):
        super().__init__()

        self.band_groups = params.CHANNEL_GROUPS

        # this is used for the channel embedding: get indices of channel groups
        self.band_group_to_idx = {
            group_name: data['id'] for group_name,data in self.band_groups.items()
        }
        # index of Dynamic World as last channel group index
        self.band_group_to_idx['DW'] = max(self.band_group_to_idx.values()) + 1

        # linear projection from encoder output to decoder embedding
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # learnable mask token for masked time steps
        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        # temporal decoder attention block
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # per channel group a linear layer which projects from embedding dim to 
        # number of channels of channel group
        self.eo_decoder_pred = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embed_dim, len(data['idx']))
                for group_name, data in self.band_groups.items()
            }
        )
        # separate layer for Dynamic World 
        self.dw_decoder_pred = nn.Linear(decoder_embed_dim, params.DW_CLASSES)

        # the positional, monthly, channel embedding
        self.channel_embeddings = channel_embeddings
        channel_embedding_dims = channel_embeddings.weight.shape[-1]
        remaining_embeddings = decoder_embed_dim - channel_embedding_dims

        self.max_sequence_length = max_sequence_length
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, int(remaining_embeddings) // 2),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(remaining_embeddings) // 2)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

    def initialize_weights(self):
        '''
        Initializes embedding weights and model weights via self.apply. 
        '''
        
        # initializes position sinusoidal encoding 
        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        # copy values into position embedding
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        Initializes the encoder weights via self.apply with Xavier-uniform
        for linear layers, bias of linear layer as 0.0. Initializes layer norm
        with 1.0 and bias of layer norm with 0.0.
        '''
        
        # linear layer
        if isinstance(m, nn.Linear):
            # use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # layer norm
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_masked_tokens(self, x, orig_indices, x_mask):
        '''
        Fills cropped time dimension with mask tokens to obtain original length
        before compression and brings time steps back into original order.
        '''

        mask_token=self.mask_token.to(x.dtype)
        all_masked = repeat(mask_token, "d -> b t d", b=x.shape[0], t=orig_indices.shape[1])
        mask = torch.cat(
            (
                x_mask,
                torch.ones((x.shape[0], orig_indices.shape[1] - x.shape[1]), device=params.DEVICE),
            ),
            dim=-1,
        )
        # can't set value on leaf variable
        out = all_masked.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[~mask.bool()] = x[~x_mask.bool()]
        # then move them to their original positions
        out = out.scatter(1, orig_indices[:, :, None].expand_as(out), out)
        return out

    def add_embeddings(self, x, month: Union[torch.Tensor, int]):
        '''
        Method adds month embedding, channel group embedding and position embedding
        per token to decoder input x. 
        '''

        num_channel_groups = len(self.band_group_to_idx)
        # time steps -2 since srtm and latlon are removed and each possess only one time step,
        # and -1 since the srtm channel group doesn't have timesteps
        num_timesteps = int((x.shape[1] - 2) / (num_channel_groups - 1))
        # get index of srtm
        srtm_index = self.band_group_to_idx['SRTM'] * num_timesteps
        # month index table
        months = month_to_tensor(month, x.shape[0], num_timesteps)

        # when expanding the encodings, each channel_group gets num_timesteps
        # encodings. However, there is only one SRTM token so the excess SRTM 
        # encodings are removed
        remove_mask = torch.full(size=(num_timesteps * num_channel_groups,), fill_value=False)
        remove_mask[torch.arange(num_timesteps - 1) + srtm_index] = True

        # month embedding
        month_embedding = repeat(
            self.month_embed(months), "b t d -> b (repeat t) d", repeat=num_channel_groups
        )
        month_embedding = month_embedding[:, ~remove_mask]
        month_embedding[:, srtm_index] = 0

        # sinusoidal position embedding
        positional_embedding = repeat(
            self.pos_embed[:, :num_timesteps, :],
            "b t d -> (b2 b) (t2 t) d",
            b2=x.shape[0],
            t2=num_channel_groups,
        )
        positional_embedding = positional_embedding[:, ~remove_mask]
        positional_embedding[:, srtm_index] = 0

        # channel embedding
        channel_embeddings = torch.repeat_interleave(
            self.channel_embeddings.weight, repeats=num_timesteps, dim=0
        )
        channel_embeddings = repeat(channel_embeddings, "c d -> b c d", b=x.shape[0])
        channel_embeddings = channel_embeddings[:, ~remove_mask]

        # concatenation of month, channel group, position embedding
        positional_embedding = torch.cat(
            (month_embedding, channel_embeddings, positional_embedding), dim=-1
        )

        # add the zero embedding for the latlon token
        positional_embedding = torch.cat(
            [torch.zeros_like(positional_embedding[:, 0:1, :]), positional_embedding], dim=1
        )

        # add to input
        x += positional_embedding
        return x

    def reconstruct_inputs(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Reconstructs EO channels per channel group from decoder tokens and 
        logits for Dynamic World tokens. 
        '''

        # remove the latlon token
        x = x[:, 1:, :]

        # split into channel groups
        num_channel_groups = len(self.band_group_to_idx) - 1
        # reconstruct original number of time steps per channel group
        num_timesteps = int((x.shape[1] - 1) / num_channel_groups)
        # indices of srtm
        srtm_index = self.band_group_to_idx['SRTM'] * num_timesteps
        # extract srtm
        srtm_token = x[:, srtm_index : srtm_index + 1, :]
        
        # remove srtm from input
        mask = torch.full((x.shape[1],), True, device=x.device)
        mask[torch.tensor(srtm_index)] = False
        x = x[:, mask]

        # reshape  to channel groups
        x = x.view(x.shape[0], num_channel_groups, num_timesteps, x.shape[-1])

        # reconstructs channels per channel group as decoder prediction
        eo_output, dw_output = [], None
        for group_name, idx in self.band_group_to_idx.items():
            if group_name == 'SRTM':
                eo_output.append(
                    repeat(
                        self.eo_decoder_pred[group_name](srtm_token),
                        "b t d -> b (t2 t) d",
                        t2=num_timesteps,
                    )
                )
            else:
                # srtm is removed, so all indices shifted by -1
                if idx > self.band_group_to_idx['SRTM']:
                    idx -= 1
                group_tokens = x[:, idx]
                if group_name == "DW":
                    dw_output = self.dw_decoder_pred(group_tokens)
                else:
                    eo_output.append(self.eo_decoder_pred[group_name](group_tokens))

        # we can just do this concatenation because the BANDS_GROUP_IDX
        # is ordered
        return torch.cat(eo_output, dim=-1), cast(torch.Tensor, dw_output)

    def forward(
        self, 
        x, 
        orig_indices, 
        x_mask, 
        month
    ):
        '''
        Conducts forward pass of the decoder: has to be implemented in sub-classes.
        '''

        raise NotImplementedError
