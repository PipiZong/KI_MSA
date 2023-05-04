import torch
import numpy as np
import random
import _pickle as pk
import config_adapter as default_config
from torch.utils.data import Dataset, DataLoader



class MOSEIDataset(Dataset):
    def __init__(self, type, config=default_config):
        raw_data_path = config.MOSI.path.raw_data_path_mosei
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[type]

        if 'audio_lengths' not in self.data.keys():
            audio_len = self.data['audio'].shape[1]
            self.data['audio_lengths'] = [audio_len] * self.data['audio'].shape[0]
        if 'vision_lengths' not in self.data.keys():
            vision_len = self.data['vision'].shape[1]
            self.data['vision_lengths'] = [vision_len] * self.data['vision'].shape[0]


        self.data['raw_text'] = np.array(self.data['raw_text'])
        self.data['id'] = np.array(self.data['id'])
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
        self.scaled_embedding_averaged = False
        # self.__normalize()
        self.__gen_mask()


    def __gen_mask(self):
        vision_tmp = torch.sum(torch.tensor(self.data['vision']), dim=-1)
        vision_mask = (vision_tmp == 0)

        for i in range(self.size):
            vision_mask[i][0] = False
        vision_mask = torch.cat((vision_mask[:, 0:1], vision_mask), dim=-1)

        self.data['vision_padding_mask'] = vision_mask
        audio_tmp = torch.sum(torch.tensor(self.data['audio']), dim=-1)
        audio_mask = (audio_tmp == 0)
        for i in range(self.size):
            audio_mask[i][0] = False
        audio_mask = torch.cat((audio_mask[:, 0:1], audio_mask), dim=-1)
        self.data['audio_padding_mask'] = audio_mask

    def __pad(self):

        PAD = torch.zeros(self.data['vision'].shape[0], 1, self.data['vision'].shape[2])
        self.data['vision'] = np.concatenate((self.data['vision'], PAD), axis=1)
        Ones = torch.ones(self.data['vision'].shape[0], self.data['vision'].shape[2])
        for i in range(len(self.data['vision'])):
            self.data['vision'][i, self.data['vision_lengths'], :] = Ones

        PAD = torch.zeros(self.data['audio'].shape[0], 1, self.data['audio'].shape[2])
        self.data['audio'] = np.concatenate((self.data['audio'], PAD), axis=1)
        Ones = torch.ones(self.data['audio'].shape[0], self.data['audio'].shape[2])
        for i in range(len(self.data['audio'])):
            self.data['audio'][i, self.data['audio_lengths'], :] = Ones

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        for key in self.data:
            samples[key] = self.data[key][idx]
        return samples


def MOSEIDataloader(name, batch_size=None,
                    shuffle=True,
                    num_workers=0,
                    prefetch_factor=2,
                    config=default_config):
    if batch_size is None:
        print('batch size not defined')
        return
    dataset = MOSEIDataset(name)
    sampler = None
    drop_last = False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=drop_last, timeout=0,
                      worker_init_fn=None, prefetch_factor=prefetch_factor,
                      persistent_workers=False)
