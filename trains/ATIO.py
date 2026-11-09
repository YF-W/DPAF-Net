from .singleTask.BASE import BaseModel

class ATIO:
    def getTrain(self, args):
        if args.model_name != 'base':
            raise ValueError('This repository provides the base DPAF-Net trainer.')
        return BaseModel(args)
