from model.text_encoder import TextEncoder
from model.vision_encoder import VisionEncoder
from model.audio_encoder import AudioEncoder
from model.adapter_fusion import af
import torch
import config_adapter as default_config
from torch import nn
import torch.nn.functional as F
from model.classifier import BaseClassifier
from util.common import check_dir
from transformers.models.bert.modeling_bert import BertEncoder
from model.contrast_6 import SupConLoss


class projector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(projector, self).__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class SEGating(nn.Module):
    def __init__(self, input_dim):
        super(SEGating, self).__init__()
        self.gating = nn.Sequential(
            nn.Linear(input_dim, input_dim//8),
            nn.BatchNorm1d(input_dim//8),
            nn.ReLU(),
            nn.Linear(input_dim//8, input_dim),
            nn.Sigmoid()
        )

    def forward(self, emb):
        mask = self.gating(emb)
        emb = emb * mask
        return emb


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Adapter(nn.Module):
    def __init__(self, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.adapter_config.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.adapter_config.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()



class TVA_fusion(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, drop_out=None, config=default_config):
        super(TVA_fusion, self).__init__()
        self.config = config

        self.pj_v = nn.Linear(35, config.MOSI.downStream.vision_fea_dim)
        self.pj_a = nn.Linear(74, config.MOSI.downStream.audio_fea_dim)

        self.text_encoder = TextEncoder(name=name, with_projector=False, config=config)
        self.adapter_t = nn.ModuleList([Adapter(config.Adapter) for _ in range(len(config.Adapter.adapter_list_t))])
        self.pooler_t = BertPooler(config.Adapter)
        self.com_dense_t = nn.Linear(self.config.Adapter.hidden_size * 2, self.config.Adapter.hidden_size)

        self.vision_encoder = VisionEncoder(config=config)
        self.adapter_v = nn.ModuleList([Adapter(config.Adapter) for _ in range(len(config.Adapter.adapter_list_v))])
        self.ln_v = nn.LayerNorm(self.config.Adapter.hidden_size)
        self.com_dense_v = nn.Linear(self.config.Adapter.hidden_size * 2, self.config.Adapter.hidden_size)

        self.audio_encoder = AudioEncoder(config=config)
        self.adapter_a = nn.ModuleList([Adapter(config.Adapter) for _ in range(len(config.Adapter.adapter_list_a))])
        self.ln_a = nn.LayerNorm(self.config.Adapter.hidden_size)
        self.com_dense_a = nn.Linear(self.config.Adapter.hidden_size * 2, self.config.Adapter.hidden_size)
        if self.config.MOSI.downStream.TVAExp_fusion.segating:
            self.fusion_mod = SEGating(self.config.Adapter.hidden_size * 3)
        self.fusion_dropout = nn.Dropout(p=config.MOSI.downStream.TVAExp_fusion.post_fusion_dropout)
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.MOSI.downStream.text_drop_out

        hidden_size = [encoder_fea_dim, int(encoder_fea_dim / 2), int(encoder_fea_dim / 4), int(encoder_fea_dim / 8),
                       ]

        self.TVA_decoder = BaseClassifier(input_size=encoder_fea_dim * 3,
                                          hidden_size=hidden_size,
                                          output_size=1, drop_out=drop_out,
                                          name='TVARegClassifier', )

        self.mono_decoder = BaseClassifier(input_size=encoder_fea_dim,
                                           hidden_size=hidden_size[1:],
                                           output_size=1, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )
        self.contrast = SupConLoss(temperature=0.07)

        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.model_path = config.MOSI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)

        self.batch_size = config.MOSI.downStream.TVAExp_fusion.batch_size
        self.set_train()

    def forward(self, sample1, return_loss=True, return_emb=False, device=None):
        if device is None:
            device = self.device

        text1 = sample1['raw_text']
        vision1 = sample1['vision'].clone().detach().to(device).float()
        audio1 = sample1['audio'].clone().detach().to(device).float()
        #########
        if vision1.shape[-1] != self.config.MOSI.downStream.vision_fea_dim:
            vision1 = self.pj_v(vision1)
        if audio1.shape[-1] != self.config.MOSI.downStream.audio_fea_dim:
            audio1 = self.pj_a(audio1)[:, :375, :] ##############

        label1 = sample1['regression_labels'].clone().detach().to(device).float()
        label_T1 = sample1['regression_labels'].clone().detach().to(device).float()
        label_V1 = sample1['regression_labels'].clone().detach().to(device).float()
        label_A1 = sample1['regression_labels'].clone().detach().to(device).float()
        key_padding_mask_V1, key_padding_mask_A1 = (sample1['vision_padding_mask'].clone().detach().to(device),
                                                    sample1['audio_padding_mask'].clone().detach().to(device))
        key_padding_mask_A1 = key_padding_mask_A1[:, :376]

        x_t, t_hidden_states, t_output = self.text_encoder(text1, device=device, output_hidden_states=True)
        hidden_states_last = af(self.config.Adapter.adapter_list_t, self.config.Adapter.adapter_skip_layers, t_hidden_states, t_output, self.adapter_t)
        hidden_states_last_t = self.pooler_t(hidden_states_last)
        x_t_embed = self.com_dense_t(torch.cat([x_t, hidden_states_last_t],dim=1))


        x_v, v_hidden_states, v_output = self.vision_encoder(vision1, key_padding_mask=key_padding_mask_V1, device=device, output_hidden_states=True)
        hidden_states_last = af(self.config.Adapter.adapter_list_v, self.config.Adapter.adapter_skip_layers,
                                v_hidden_states, v_output, self.adapter_v)
        hidden_states_last = self.ln_v(hidden_states_last)
        hidden_states_last_v = torch.mean(hidden_states_last, dim=-2, keepdim=True)
        x_v_embed = self.com_dense_v(torch.cat([x_v, hidden_states_last_v], dim=2))
        x_v = x_v.squeeze(1)
        hidden_states_last_v = hidden_states_last_v.squeeze(1)
        x_v_embed = x_v_embed.squeeze(1)

        x_a, a_hidden_states, a_output = self.audio_encoder(audio1, key_padding_mask=key_padding_mask_A1, device=device, output_hidden_states=True)
        hidden_states_last = af(self.config.Adapter.adapter_list_a, self.config.Adapter.adapter_skip_layers,
                                a_hidden_states, a_output, self.adapter_a)
        hidden_states_last = self.ln_a(hidden_states_last)
        hidden_states_last_a = torch.mean(hidden_states_last, dim=-2, keepdim=True)
        x_a_embed = self.com_dense_a(torch.cat([x_a, hidden_states_last_a], dim=2))
        x_a_embed = x_a_embed.squeeze(1)
        x_a = x_a.squeeze(1)
        hidden_states_last_a = hidden_states_last_a.squeeze(1)

        x1 = torch.cat((x_t_embed, x_v_embed, x_a_embed), dim=-1)
        if self.config.MOSI.downStream.TVAExp_fusion.segating:
            x1 = self.fusion_mod(x1)
        x1 = self.fusion_dropout(x1)
        x1_mono = torch.cat((x_t_embed, x_v_embed, x_a_embed), dim=0)
        label1_mono = torch.cat((label_T1, label_V1, label_A1), dim=0)



        if return_loss:
            pred = self.TVA_decoder(x1)
            pred_mono = self.mono_decoder(x1_mono)
            pred_loss = self.criterion(pred.squeeze(), label1)
            mono_task_loss = self.criterion(pred_mono.squeeze(), label1_mono)

            #################################ADD CL loss
            features = torch.cat((F.normalize(x_t, dim=1).unsqueeze(1), F.normalize(x_v, dim=1).unsqueeze(1),
                                  F.normalize(x_a, dim=1).unsqueeze(1), F.normalize(hidden_states_last_t, dim=1).unsqueeze(1),
                                  F.normalize(hidden_states_last_v, dim=1).unsqueeze(1), F.normalize(hidden_states_last_a, dim=1).unsqueeze(1),
                                  ), dim=1)
            labels = torch.round(label1)
            cl_loss = self.contrast(features, labels)
            ###############################################

            loss = pred_loss + self.config.MOSI.downStream.lamda * mono_task_loss + 0.01 * cl_loss
            if return_emb:
                return pred, x1, loss, pred_loss
            else:
                return pred, (x_t_embed, x_v_embed, x_a_embed), loss, pred_loss
        else:
            if return_emb:
                return (x_t_embed, x_v_embed, x_a_embed)

    def save_model(self, name):
        mode_path = self.model_path + name + '.ckpt'

        print('model saved at:')
        print(mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name, load_pretrain=False):
        if load_pretrain:
            mode_path = self.model_path + name + '.ckpt'
            print('model loaded from:')
            print(mode_path)
            self.load_state_dict(torch.load(mode_path, map_location=self.device))

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [False, False, False, True]

        for param in self.parameters():
            param.requires_grad = train_module[3]
        self.text_encoder.set_train(train_module=train_module[0:2])
        self.vision_encoder.set_train(train_module=train_module[2])
        self.audio_encoder.set_train(train_module=train_module[2])

