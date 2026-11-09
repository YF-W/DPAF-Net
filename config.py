import json
import os
from pathlib import Path
from easydict import EasyDict as edict

def get_config_regression(model_name, dataset_name, config_file=None, data_root=None):
    path = Path(config_file) if config_file else Path(__file__).parent / 'config/config_regression.json'
    with path.open(encoding='utf-8') as handle:
        config_all = json.load(handle)
    model = config_all[model_name]
    dataset = config_all['datasetCommonParams'][dataset_name]
    kind = 'aligned' if model['commonParams']['need_data_aligned'] and 'aligned' in dataset else 'unaligned'
    args = dict(model_name=model_name, dataset_name=dataset_name)
    args.update(dataset[kind])
    args.update(model['commonParams'])
    args.update(model['datasetParams'][dataset_name])
    feature = Path(args['featurePath']).expanduser()
    data_root = data_root or os.environ.get('MSA_DATA_ROOT') or config_all['datasetCommonParams'].get('dataset_root_dir') or 'data'
    args['featurePath'] = str(feature if feature.is_absolute() else Path(data_root).expanduser() / feature)
    return edict(args)
