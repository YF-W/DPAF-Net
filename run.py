import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import count_parameters, setup_seed

def main():
    parser = argparse.ArgumentParser(description='DPAF-Net training and evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--model_name', choices=['base'], default='base')
    parser.add_argument('--dataset_name', choices=['mosi', 'mosei', 'sims', 'simsv2'], default='mosi')
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', default='.')
    cli = parser.parse_args()
    if cli.epochs < 1 or cli.num_workers < 0:
        parser.error('epochs must be positive and num_workers must be non-negative')
    if cli.mode == 'test' and not cli.checkpoint:
        parser.error('--mode test requires --checkpoint')
    if cli.mode == 'test' and not Path(cli.checkpoint).is_file():
        parser.error(f'Checkpoint not found: {cli.checkpoint}')

    args = get_config_regression(cli.model_name, cli.dataset_name, cli.config_file, cli.data_root)
    if not Path(args.featurePath).is_file():
        parser.error(f'Feature file not found: {args.featurePath}. Set --data_root or MSA_DATA_ROOT.')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device if cli.device == 'auto' else cli.device)
    args.update(train_mode='regression', epochs=cli.epochs, cur_seed=1, result_name='',
                custom_feature=None, feature_T=None, feature_A=None, feature_V=None)
    output = Path(cli.output_dir)
    args.model_save_dir = output / 'saved_models' / cli.dataset_name
    args.model_save_path = args.model_save_dir / f'base_{cli.dataset_name}.pth'
    result_dir = output / 'results' / 'base'
    log_dir = output / 'logs'
    for path in [args.model_save_dir, result_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logger = logging.getLogger('MSA')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(log_dir / f'base-{cli.dataset_name}-{cli.mode}-{stamp}.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    setup_seed(cli.seed)
    dataloader = MMDataLoader(args, cli.num_workers)
    model = AMIO(args).to(args.device)
    trainer = ATIO().getTrain(args)
    logger.info('Model: DPAF-Net; dataset: %s; seed: %s; parameters: %s', cli.dataset_name, cli.seed, count_parameters(model))
    if cli.mode == 'train':
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=True)
        rows = []
        for epoch, valid in enumerate(epoch_results['valid']):
            row = {'epoch': epoch + 1}
            row.update({f'valid_{key}': value for key, value in valid.items() if key != 'Loss'})
            row['Train_Loss'] = epoch_results['train'][epoch]['Loss']
            row['Val_Loss'] = valid['Loss']
            rows.append(row)
        csv_path = result_dir / f'BASE_{cli.dataset_name}.csv'
        if csv_path.exists():
            csv_path = result_dir / f'BASE_{cli.dataset_name}_{stamp}.csv'
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        checkpoint = args.model_save_path
    else:
        checkpoint = Path(cli.checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(args.device)
    setup_seed(cli.seed)
    metrics = trainer.do_test(model, dataloader['test'], mode='TEST')
    report = {'model': 'DPAF-Net', 'dataset': cli.dataset_name, 'seed': cli.seed,
              'checkpoint': str(checkpoint), 'metrics': {key: float(value) for key, value in metrics.items()}}
    report_path = result_dir / f'test_{cli.dataset_name}_{stamp}.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    logger.info('Test results saved to %s', report_path)
    print(json.dumps(report, indent=2))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

if __name__ == '__main__':
    main()
