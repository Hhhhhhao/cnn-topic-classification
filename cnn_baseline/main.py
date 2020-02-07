import os
import sys
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(WORKING_DIR)
import torch
import numpy as np
import argparse
import random
# fix random seeds for reproducibility
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
from trainers import CelebATrainer
from evaluator import Evaluator
from utils.config import process_config


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default=os.path.join(WORKING_DIR, 'src/configs/classification.yaml'),
        help='Path of configuration file in yaml format')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    config.data_dir = os.path.join(WORKING_DIR, config.data_dir)

    # Train the model
    if 'celeba' in config.exp_name:
        trainer = CelebATrainer(config)
    else:
        raise NotImplementedError
    trainer.train()

    # Evaluate the model
    evaluator = Evaluator(config, config.checkpoint_dir, config.resume_epoch)
    evaluator.test()


if __name__ == '__main__':
    main()
