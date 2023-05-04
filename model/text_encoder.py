from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
from model.projector import FeatureProjector
from torch import nn
import config_adapter as default_config


class TextEncoder(nn.Module):
    def __init__(self, name=None, fea_size=None, proj_fea_dim=None, drop_out=None, with_projector=True,
                 config=default_config):
        super(TextEncoder, self).__init__()
        self.name = name
        if fea_size is None:
            fea_size = config.MOSI.downStream.text_fea_dim
        if proj_fea_dim is None:
            proj_fea_dim = config.MOSI.downStream.proj_fea_dim
        if drop_out is None:
            drop_out = config.MOSI.downStream.text_drop_out
        if config.USEROBERTA:
            self.tokenizer = BertTokenizer.from_pretrained("roberta-base")
            self.extractor = BertModel.from_pretrained("roberta-base")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.extractor = BertModel.from_pretrained('bert-base-uncased')
        self.with_projector = with_projector
        if with_projector:
            self.projector = FeatureProjector(fea_size, proj_fea_dim, drop_out=drop_out,
                                              name='text_projector', config=config)
        self.device = config.DEVICE

    def forward(self, text, device=None, output_hidden_states=False):
        if device is None:
            device = self.device

        x = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
            device)
        xx = self.extractor(**x, output_hidden_states=output_hidden_states)
        x = xx['pooler_output']
        if self.with_projector:
            x = self.projector(x)
        if output_hidden_states==True:
            return (x, xx.hidden_states, xx.last_hidden_state)
        else:
            return x

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        for name, param in self.extractor.named_parameters():
                param.requires_grad = train_module[0]

        if self.with_projector:
            for param in self.projector.parameters():
                param.requires_grad = train_module[1]
