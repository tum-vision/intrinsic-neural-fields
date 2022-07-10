import numpy as np
import torch
import argparse
import random
import os

from config import load_config_file, get_data, get_model_and_optim, get_loss_fn, get_renderer, get_seed
from trainer import Trainer
from mesh import load_mesh
from utils import model_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument('--allow_checkpoint_loading', default=False, action="store_true")
    parser.add_argument('--data_parallel', default=False, action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = load_config_file(args.config_path, args.allow_checkpoint_loading)

    seed = get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    mesh = load_mesh(config["data"]["mesh_path"])
    data = get_data(config, device)

    model, optim = get_model_and_optim(config, mesh, device)

    # Print model summary
    model_summary(model, data)
    
    if args.data_parallel:
        device_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    loss_fn = get_loss_fn(config)
    renderer = get_renderer(config, model, mesh, device)

    # Seed again because different model architectures change seed. Make train samples consistent.
    # https://discuss.pytorch.org/t/shuffle-issue-in-dataloader-how-to-get-the-same-data-shuffle-results-with-fixed-seed-but-different-network/45357/9
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(model, optim, loss_fn, renderer, data, mesh, config, device)
    trainer.train()


if __name__ == "__main__":
    main()
