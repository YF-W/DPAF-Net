import torch.nn as nn
from .singleTask.BASE import BaseModel
from .subNets.AlignNets import AlignSubNet

class AMIO(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model_name != 'base':
            raise ValueError('This repository provides the base DPAF-Net model.')
        self.need_model_aligned = args.get('need_model_aligned', None)
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args:
                args['seq_lens'] = self.alignNet.get_seq_len()
        self.Model = BaseModel(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
