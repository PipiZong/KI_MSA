import torch
import torch.nn as nn
import config as default_config



class PositionEncodingTraining(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, fea_size=None, tf_hidden_dim=None, drop_out=None, config=default_config):
        super().__init__()
        if fea_size is None:
            fea_size = config.MOSI.downStream.vision_fea_dim
        if tf_hidden_dim is None:
            tf_hidden_dim = config.MOSI.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.MOSI.downStream.vision_drop_out

        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        num_patches = 500
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TfEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu',
                 config=default_config):
        super(TfEncoder, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')

        self.device = config.DEVICE
        self.model_type = 'vision_encoder'
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining()

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None, output_hidden_states=False):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        if output_hidden_states==True:
            output, hidden_states = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask, output_hidden_states=output_hidden_states)
            hidden_states = tuple(i.transpose(0, 1) for i in hidden_states)
            return output.transpose(0, 1), hidden_states
        else:
            output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)
            return output.transpose(0, 1)


class VisionEncoder(nn.Module):
    def __init__(self, name=None, fea_size=None, encoder_fea_dim=None, nhead=None, dim_feedforward=None,
                 num_layers=None,
                 drop_out=0.5, config=default_config):
        super(VisionEncoder, self).__init__()
        self.name = name
        if fea_size is None:
            fea_size = config.MOSI.downStream.vision_fea_dim
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        if nhead is None:
            nhead = config.MOSI.downStream.vision_nhead
        if drop_out is None:
            drop_out = config.MOSI.downStream.vision_drop_out
        if dim_feedforward is None:
            dim_feedforward = config.MOSI.downStream.encoder_fea_dim
        if num_layers is None:
            num_layers = config.MOSI.downStream.vision_tf_num_layers

        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(d_model=encoder_fea_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                 num_layers=num_layers,
                                 dropout=drop_out, activation='gelu',
                                 config=config)

        self.device = config.DEVICE
        self.encoder.device = self.device
        self.activation = nn.Tanh()
        self.cls_embedding = nn.Parameter()
        self.layernorm = nn.LayerNorm(encoder_fea_dim)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)
        # self.fc = nn.Linear(709,768)

    def forward(self, vision, key_padding_mask, device=None, output_hidden_states=False):
        if device is None:
            device = self.device

        if output_hidden_states == True:
            xx, hidden_states = self.encoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask, output_hidden_states=output_hidden_states)
            x = self.layernorm(xx)
            x = torch.mean(x, dim=-2, keepdim=True)

            return x, hidden_states, xx

        else:
            x = self.encoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask)
            x = self.layernorm(x)
            x = torch.mean(x, dim=-2, keepdim=True)

            return x

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = True
        for param in self.parameters():
            param.requires_grad = train_module


