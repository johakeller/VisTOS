'''
VisTOS attention-based spatial encoding model based on the Presto architecture 
(Tseng et al., 2024). Can generate a pretraining model, consisting of an encoder
and a decoder (autoencoder) and a fine-tuning model, consisting of an encoder and
a fine-tuning head. Basic model elements are imported from module vistos_base.
'''

import os
from copy import deepcopy

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

import params
from model import vistos_base


class Encoder(vistos_base.EncoderBase,nn.Module):
    '''
    Encoder module of the VisTOS autoencoder with attention-based spatial encoding. 
    Performs pixel-wise positional embedding of the model input (multi-channel pixel 
    time series in sequential order). Subsequently, the input is passed through the 
    first attention block, which performs a temporal self-attention encoding. After 
    that, the input is unfolded into overlapping visual fields via unfold and padded. 
    Then a pixel-wise positional encoding is added, to mark a pixel time series' spatial
    position in a visual field. Then, with the second attention block, spatial self-
    attention is performed. For fine-tuning, the time dimension is aggregated via a 
    set of hierarchical convolutions and a GELU activation function.
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
        dropout=params.DROPOUT,
    ):
        super().__init__(
            embedding_size=embedding_size,
            channel_embed_ratio=channel_embed_ratio,
            month_embed_ratio=month_embed_ratio,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            vis_field_size=vis_field_size,
            mode=mode,
            dropout=dropout,
        )
        # unfold to visual fields 
        self.unfold=nn.Unfold(kernel_size=(self.vis_field_size,self.vis_field_size), padding=0, stride=1)
        
        # spatial encoding blocks
        self.block_2 = nn.ModuleList(
            [
                vistos_base.Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    dropout=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

        # learned pixel position encoding (1,p,1,e)
        # -> needs to be repeated over batch b and time dim t_new
        # row encoding 
        self.pix_row_embed= nn.Embedding(vis_field_size, int(embedding_size/2))
        # column encoding
        self.pix_col_embed=nn.Embedding(vis_field_size, int(embedding_size/2))
        # row tensor (flat)
        self.vis_field_rows=torch.arange(self.vis_field_size).repeat_interleave(self.vis_field_size).to(params.DEVICE)
        # column tensor (flat)
        self.vis_field_cols=torch.arange(self.vis_field_size).repeat(self.vis_field_size).to(params.DEVICE)      
        
        # initialize encoder weights
        self.initialize_weights()

    def forward(
        self,
        input_dict: dict,
        eval_task: bool = True,
    ):
        '''
        Conducts forward pass of encoder: 1) pixel-wise embedding of the input; 
        2) temporal self-attention encoding; 3) unfolding to visual fields in the 
        spatial dimension and padding; 4) pixel-position encoding of visual field;
        5) spatial self-attention; 6) optional aggregation of time dimension in 
        fine-tuning mode. Implementation of the abstract method from class EncoderBase.
        '''

        # extract input data from input dict
        x=input_dict['EO']
        dw=input_dict['DW'].long()
        latlons=input_dict['loc']
        # None if no mask, dw_mask not used
        mask=input_dict.get('EO_mask', None)
        month=input_dict.get('month',0)

        # basic dimensions: batch size b
        b= x.shape[0]
        # width of image
        w=self.img_width 
        # height: how many pixel rows 
        h=int(b/w)  
        # num pixels of visual field (p)
        p=self.vis_field_size**2

        # init mask with 0 in input shape if None is passed
        if mask is None:
            mask = torch.zeros_like(x, device=x.device).float()

        # 0. PIXEL-WISE EMBEDDING: linear embedding of tokens (channel group, time step) 
        # + (month embedding, channel embedding, position embedding)
        # month embedding: input tensor of month indices in [0,11] 
        # -> create embedding of dimensions (b, t, month_embedding_size)
        months = vistos_base.month_to_tensor(month, x.shape[0], x.shape[1])
        month_embedding = self.month_embed(months)
        positional_embedding = repeat(
            self.pos_embed[:, : x.shape[1], :], "b t d -> (repeat b) t d", repeat=x.shape[0]
        )

        # Tseng et al. (2024) assume the number of masked patches is the same for all items in 
        # the batch. Otherwise things become a headache
        all_tokens, all_masks = [], []
        for channel_group, data in params.CHANNEL_GROUPS.items():
            # identify indices of channels belonging to channel_group
            idx=data['idx']
            # token embedding of the (channel group, time step) tokens
            tokens = self.eo_patch_embed[channel_group](x[:, :, idx])
            # linear embedding of the channel group index
            channel_embedding = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(params.DEVICE)
            )
            # repeat over the entire batch
            channel_embedding = repeat(channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1])
            # for SRTM, the embedding is reduced to a single token instead of a token per timestep: 
            # the entire positional embedding has dimensions (b, 1, e) for SRTM (assumed to be time-invariant)
            if channel_group == "SRTM":
                channel_wise_positional_embedding = torch.cat(
                    (
                        # month embedding (b, 1, month_embedding_size)
                        torch.zeros_like(month_embedding[:, 0:1]),
                        # channel embedding (b, 1, channel_embedding_size)
                        channel_embedding[:, 0:1],
                        # positional embedding (b, 1, position_embedding_size)
                        torch.zeros_like(positional_embedding[:, 0:1]),
                    ),
                    dim=-1,
                )
                # just one time step remains
                indices = slice(0, 1)
            else:
                # time dependent input channels: 
                # concatenate month embedding, channel embedding, positional embedding via time dimension
                channel_wise_positional_embedding = torch.cat(
                    (month_embedding, channel_embedding, positional_embedding), dim=-1
                )
                # all time steps remain
                indices = slice(None)

            # slice the tokens
            tokens = tokens[:, indices]
            # add channel_wise_positional embedding to token embedding 
            tokens += channel_wise_positional_embedding
            # append channel group to embedded tokens list 
            all_tokens.append(tokens)

            # mask for group: 1 if time step in group is masked
            group_mask = torch.max(mask[:, indices, idx], dim=-1)[0]
            all_masks.append(group_mask)

        # then, Dynamic World (separate embedding)
        tokens = self.dw_embed(dw)
        # channel embedding for Dynamic World
        channel_embedding = self.channel_embed(
            torch.tensor(self.band_group_to_idx['DW']).long().to(params.DEVICE)
        )
        channel_embedding = repeat(channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1])
        # positional embedding for Dynamic World: concatenation of 
        # month embedding, channel embedding, position embedding 
        positional_embedding = torch.cat(
            (month_embedding, channel_embedding, positional_embedding), dim=-1
        )
        # add positional embedding to token embedding
        tokens += positional_embedding
        all_tokens.append(tokens)

        # now calculate the mask for these [b, t] tokens
        group_mask = dw == params.DW_CLASSES
        all_masks.append(group_mask)

        x = torch.cat(all_tokens, dim=1)  # (b, t, e)
        mask = torch.cat(all_masks, dim=1)  # (b,t)
        # apply compression of time steps
        x, orig_indices, upd_mask = self.mask_tokens(x, mask)

        # append latlon tokens
        latlon_tokens = self.latlon_embed(self.cartesian(latlons)).unsqueeze(1)
        x = torch.cat((latlon_tokens, x), dim=1)
        upd_mask = torch.cat((torch.zeros(x.shape[0])[:, None].to(params.DEVICE), upd_mask), dim=1)
        orig_indices = torch.cat(
            (torch.zeros(x.shape[0])[:, None].to(params.DEVICE).int(), orig_indices + 1),
            dim=1,
        )

        # shape of the embedded batch (b,t_new,e=embedding_size)
        _, t_new, e= x.shape

        # 1. TEMPORAL FACTORIZATION
        for blk in self.block_1:
            x = blk(x, attn_mask=~upd_mask.bool())
        # if visual field size ==1: skip spatial block and return x, orig_indices and mask
        if self.vis_field_size ==1:
            # mask will be a boolean of shape [batch, total_num_tokens]
            if eval_task:
                # set masked tokens to 0
                x = x * (1 - upd_mask.unsqueeze(-1))
                # reduce time dimension via learned linear layer (b, e, t_new)
                x = self.adapt_pool(x.permute(0,2,1))
                # (b, e, t_new//2)
                x=self.merge_time_non_linear(self.merge_time_1(x))
                # (b, e,1) -> (b,e)
                x=self.merge_time_2(x).squeeze(-1)
                # (b,e) -> (b,e,1,1)
                return self.norm(x).unsqueeze(-1).unsqueeze(-1)
            return self.norm(x), orig_indices, upd_mask

        # 2. SPATIAL FACTORIZATION
        # pixel position embedding: (p, e)-> (1,p,1,e) can be simply added to input x later
        pix_pos_embed=torch.cat([self.pix_row_embed(self.vis_field_rows),self.pix_col_embed(self.vis_field_cols)], dim=-1).unsqueeze(0).unsqueeze(2).to(params.DEVICE)
        # reshape data to 2D -> wanted (t_new, e, h,w)
        x=x.reshape(h,w,t_new,e).permute(2, 3, 0,1)
        # apply padding 
        x=F.pad(x,pad=(self.pad,self.pad,self.pad,self.pad), mode='reflect')
        # x dimension (t_new, e*p, b)-> to (b, p, t_new,e)
        x=self.unfold(x).permute(2,1,0).reshape(b,e,p,t_new).permute(0,2,3,1)
        # VISUAL FIELD PIXEL POSITION ENCODING and reshape to (b*t_new, p, e)
        x=(x+pix_pos_embed).permute(0,2,1,3).reshape(b*t_new, p, e)
        
        # reshape mask to 2D -> (t_new,h,w)
        mask_attn=upd_mask.reshape(h,w,t_new).permute(2,0,1)
        # mask_att dimension (t_new, 1, h_pad, w_pad) 
        mask_attn=mask_attn.unsqueeze(1)
        # apply padding with 0 as unmasked
        mask_attn=F.pad(mask_attn,pad=(self.pad,self.pad,self.pad,self.pad), mode='constant', value=0)
        # attn_mask dimension (t_new,1*p, b)-> (b*t_new, p)
        mask_attn=self.unfold(mask_attn).permute(2,0,1).reshape(b*t_new, p)
        # spatial attention 
        for blk in self.block_2:
            x = blk(x, attn_mask=~mask_attn.bool())
        # (b,t_new,p,e)
        x=x.reshape(b,t_new,p,e)
        # attn_mask to (b, t_new, p)
        mask_attn=mask_attn.reshape(b,t_new, p)

        # if fine-tuning setting -> aggregate time dimension
        if eval_task:
            # set masked tokens to 0 -> (b, t_new,p, e)
            x = x * (1 - mask_attn.unsqueeze(-1))
            # (b, t_new, p, e)-> (b*p, t_new, e)
            x=x.permute(0,2,1,3).reshape(b*p,t_new, e)
            # reduce time dimension via learned linear layer (b*p, e, t_new')
            x = self.adapt_pool(x.permute(0,2,1))
            # (b*p, e, t_new'//2)
            x=self.merge_time_non_linear(self.merge_time_1(x))
            # (b*p, e,1) -> (b,v,v,e)
            x=self.merge_time_2(x).squeeze(-1).reshape(b,self.vis_field_size,self.vis_field_size,e)
            x=self.norm(x)
            #(b,v,v,e)->(b,e,v,v)
            return x.permute(0,3,1,2)
        return self.norm(x), orig_indices, upd_mask

class Decoder(vistos_base.DecoderBase, nn.Module):
    '''
    Decoder module used for the attention-based spatial encoding VisTOS pretraining 
    autoencoder architecture. The decoder reconstructs from the encoder representation 
    the Earth observation data and the Dynamic World data. It uses a convolutional 
    layer to aggregate the attention-produced visual fields. Inherits from the base
    class DecoderBase in the vistos_base module. 
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
        vis_field_size:int=params.VIS_FIELDS[0],
        dropout=params.DROPOUT,
    ):
        super().__init__(
            channel_embeddings=channel_embeddings,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
        )
        # set visual field size for convolution at the beginning of forward() method
        self.vis_field_size=vis_field_size
        # transform visual fields of input from (b,t,e,h,w) to (b,t,e,1,1)
        self.conv=nn.Conv2d(encoder_embed_dim, encoder_embed_dim, kernel_size=(vis_field_size,vis_field_size),stride=1)

        # initialize weights of the decoder
        self.initialize_weights()

    def forward(
            self, 
            x, 
            orig_indices, 
            x_mask, 
            month
        ):
        '''
        Forward pass of the decoder. Aggregates the visual fields in one pixel,
        embeds the encoder output tokens, adds the cropped mask tokens and 
        restores the original sequence order. Adds the pixel-wise positional 
        embeddings and applies temporal attention. Subsequently the original 
        channels are restored and masked tokens are predicted. Implementation
        of the abstract method forward() in DecoderBase.
        '''
        # input comes in shape (b, t_new, e) if no visual field applied
        if self.vis_field_size !=1:
            b,t_new,p,e=x.shape
            # (b,t_new,p,e)-> (b*t_new,e,h,w) 
            x =x.reshape(b*t_new,self.vis_field_size, self.vis_field_size,e).permute(0,3,1,2)
            # spatial pooling ->(b*t_new,e,1,1)->(b*t_new,e)
            x= self.conv(x).squeeze(-1).squeeze(-1)
            # (b*t_new,e)->(b,t_new,e)
            x=x.reshape(b,t_new,e)
        x = self.decoder_embed(x)
        x = self.add_masked_tokens(x, orig_indices, x_mask)
        x = self.add_embeddings(x, month)

        # apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.reconstruct_inputs(x)

class VistosTimeSeriesSeq2Seq(nn.Module):
    '''
    Sequence-to-sequence architecture which comprises an encoder and 
    a decoder for pretraining. This autoencoder with attention-based 
    spatial encoding architecture can be initiated with the method 
    construct(). Moreover, with the method 
    construct_finetuning_model(), a fine-tuning model with a 
    fine-tuning head instead of the decoder is constructed. 
    '''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def forward(
        self,
        x: dict
    ) -> torch.Tensor:
        '''
        Forward pass of the Seq2Seq attention-based spatial encoding
        model. Pass the input to the encoder and subsequently to the 
        decoder. 
        '''
        
        # unwrap month
        month=x['month']
        x, orig_indices, x_mask = self.encoder(
            input_dict=x,
            eval_task=False,
        )

        return self.decoder(x, orig_indices, x_mask, month)

    @classmethod
    def construct(
        cls,
        encoder_embedding_size: int = params.EMBEDDING_SIZE,
        channel_embed_ratio: float = params.CHANNEL_EMBED_RATIO,
        month_embed_ratio: float = params.MONTHS_EMBED_RATIO,
        encoder_depth: int =params.DEPTH,
        mlp_ratio: int=params.MLP_RATIO,
        encoder_num_heads:int=params.NUM_HEADS,
        decoder_embedding_size:int=params.EMBEDDING_SIZE,
        decoder_depth:int=params.DEPTH,
        decoder_num_heads:int=params.NUM_HEADS,
        max_sequence_length:int=params.MAX_SEQ_LEN,
        vis_field_size:int=params.VIS_FIELDS[0],
        dropout:float=params.DROPOUT,
    ):
        encoder = Encoder(
            embedding_size=encoder_embedding_size,
            channel_embed_ratio=channel_embed_ratio,
            month_embed_ratio=month_embed_ratio,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            num_heads=encoder_num_heads,
            max_sequence_length=max_sequence_length,
            vis_field_size=vis_field_size,
            dropout=dropout,
        )
        decoder = Decoder(
            channel_embeddings=encoder.channel_embed,
            encoder_embed_dim=encoder_embedding_size,
            decoder_embed_dim=decoder_embedding_size,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
            vis_field_size=vis_field_size,
            dropout=dropout,
        )
        return cls(encoder, decoder)

    def construct_finetuning_model(
        self,
        num_outputs: int,
        vis_field_size:int,
        img_width:int,
    ):
        '''
        Method to construct a finetuning model by passing the pre-trained encoder instance
        and adding a head for binary semantic segmentation and a head for multi-class classification.
        '''

        head1=SimpleFinetuningHead(
            num_outputs=num_outputs,
            hidden_size=self.encoder.embedding_size,
            vis_field_size=vis_field_size,
        )
                    # take encoder only from self instance, change mode to finetune for correct embedding
        model = VistosFinetuningSimple(self.encoder, head1, img_width=img_width).to(params.DEVICE)
        model.train()
        return model
    
    @classmethod
    def load_pretrained(cls, vis_field_size, dropout):
        '''
        Method to load weights from pretrained Seq2Seq model or if pretraining not finished but
        cache model dict was moved to output, load the pre-trained model from the dict.
        '''
        # create model with embedding dimensions of the pretraining dataset (12 months)
        model=cls.construct(vis_field_size=vis_field_size, dropout=dropout)
        # saved model is either model or dict if pretraining not finished and dict renamed
        saved_model=torch.load(os.path.join(params.OUTPUT, f'att_model_weights_vf{vis_field_size}.pth'), map_location=params.DEVICE, weights_only=True)
        if 'model'in saved_model:
            saved_model=saved_model['model']
        model.load_state_dict(saved_model)
        return model

class VistosFineTuningModel(nn.Module):
    '''
    Abstract class to define an attention-based spatial encoding 
    fine-tuning model.
    '''
    encoder: nn.Module

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: None,
        month: 0,
    ) -> torch.Tensor:
        '''
        Abstract forward() method of the attention-based spatial
        encoding fine-tuning model. 
        '''

        raise NotImplementedError

class SimpleFinetuningHead(nn.Module):
    '''
    Fine-tuning head which performs semantic segmentation
    linked to encoder with attention-based spatial encoding. 
    The fine-tuning head consists of a 2D convolution to 
    aggregate the visual fields of the attention-based spatial 
    encoding encoder, a GELU-activation function and a linear layer
    which outputs logit predictions for the classes.
    '''

    def __init__(
            self, 
            hidden_size:int=params.EMBEDDING_SIZE, 
            vis_field_size:int=params.VIS_FIELDS[0], 
            num_outputs:int=params.P_NUM_OUTPUTS,
            ):
        super().__init__()
        # hidden_size=embedding size of the encoder output
        self.hidden_size = hidden_size
        self.conv=nn.Conv2d(hidden_size, hidden_size, kernel_size=(vis_field_size,vis_field_size),stride=1)
        self.gelu=nn.GELU()
        # obtain num_outputs features per pixel
        self.linear=nn.Linear(hidden_size, num_outputs)
        # batch norm 
        self.batch_norm =nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        '''
        Forward method of the fine-tuning head for
        encoder with attention-based spatial encoding.
        '''

        # (b,hidden_size,h,w)->(b,hidden_size,1,1)
        x=self.gelu(self.batch_norm(self.conv(x)))
        # (b, hidden_size, 1,1)-> (b, hidden_size)
        x=x.squeeze(-1).squeeze(-1)
        # (b, num_outputs)
        return self.linear(x)

class VistosFinetuningSimple(VistosFineTuningModel):
    '''
    Fine-tuning model for semantic segmentation, comprising encoder 
    with attention-based spatial encoding and fine-tuning head. 
    Implementation of the abstract base class VistosFineTuningModel.
    '''

    def __init__(self, encoder=Encoder, head=SimpleFinetuningHead, img_width=params.P_IMG_WIDTH):
        super().__init__()
        self.encoder=deepcopy(encoder)
        # encoder not frozen for finetuning
        self.encoder.requires_grad_(True)
        # position, month, and pixel position encoding are frozen
        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)
        self.encoder.pix_row_embed.requires_grad_(False)
        self.encoder.pix_col_embed.requires_grad_(False)
        # set the correct image width in encoder
        self.encoder.set_img_width(img_width=img_width)
        self.head = head

    def forward(self, x):
        '''
        Forward method of the fine-tuning model with 
        attention-based spatial encoding. 
        '''

        x_encoder=self.encoder(x, eval_task=True)
        x= self.head(x_encoder)
        return x