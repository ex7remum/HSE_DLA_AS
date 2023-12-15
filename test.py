import torch
import os
import json
import argparse
from pathlib import Path
import gdown
import torchaudio
import hw_as.model as module_model
from hw_as.utils.parse_config import ConfigParser
from hw_as.utils import ROOT_PATH
import torch.nn.functional as F

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file, test_dir):
    os.makedirs(test_dir, exist_ok=True)

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.init_obj(config["arch"], module_model)

    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    for filename in os.listdir(test_dir):
        audio, _ = torchaudio.load(os.path.join(test_dir, filename))
        if audio.shape[1] < 64000:
            n_repeat = 64000 // audio.shape[1] + 1
            audio = audio.repeat(1, n_repeat)

        audio = audio[:, :64000].unsqueeze(0)  # 1x1x64000

        with torch.no_grad():
            logits = model(audio.to(device))["logits"]
            print(logits)
            logits = F.softmax(logits, dim=1)
            prob_spoof = logits[0][0]
            with open(out_file, 'a') as out:
                print("Probability of {} is spoof: {:.4f}\n".format(filename, prob_spoof))
                out.write("Probability of {} is spoof: {:.4f}\n".format(filename, prob_spoof))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="out.txt",
        type=str,
        help="File to write results",
    )
    args.add_argument(
        "-t",
        "--test",
        default="test_data",
        type=str,
        help="Directory with test audio",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    wg_dir = './pretrained_models'
    os.makedirs(wg_dir, exist_ok=True)

    model_url = 'https://drive.google.com/uc?export=download&id=1PSnoN-jYqfc5Z3hiftBHg5XEuNy4EPrL'
    model_path = './pretrained_models/model.pth'
    if not os.path.exists(model_path):
        print('Downloading RawNet2 model.')
        gdown.download(model_url, model_path)
        print('Downloaded RawNet2 model.')
    else:
        print('RawNet2 model already exists.')

    config_url = 'https://drive.google.com/uc?export=download&id=1F2m8wIL25_PjKq6mwkY87NeEZbeozHc9'
    config_path = './pretrained_models/config.json'
    if not os.path.exists(config_path):
        print('Downloading RawNet2 model test config.')
        gdown.download(config_url, config_path)
        print('Downloaded RawNet2 model test config.')
    else:
        print('RawNet2 model config already exists.')

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output, args.test)
