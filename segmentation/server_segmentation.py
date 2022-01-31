import logging
import socket

import click
import flwr as fl
import pandas as pd
from torch.utils.data import DataLoader
import os
from segmentation.common import *
from segmentation.data_loader import LungSegDataset
import shutil
from segmentation_models_pytorch import UnetPlusPlus

loss = []
jacc = []
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROUND = 0

BATCH_SIZE = 2
IMAGE_SIZE = 1024
MAX_ROUND = 5
CLIENTS = 3
FED_AGGREGATION_STRATEGY = 'FedAvg'
LOCAL_EPOCHS = 1
MIN_FIT_CLIENTS = 1
FRACTION_FIT = 0.3
TIME_BUDGET = 60
LEARNING_RATE = 0.001

DICE_ONLY = False
OPTIMIZER_NAME = 'Adam'

# Initialize logger
logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

strategies = {'FedAdam': fl.server.strategy.FedAdam,
              'FedAvg': fl.server.strategy.FedAvg,
              'FedYogi': fl.server.strategy.FedYogi,
              'FedAdagrad': fl.server.strategy.FedAdagrad}


def fit_config(rnd: int):
    config = {
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dice_only": DICE_ONLY,
        "optimizer_name": OPTIMIZER_NAME,
        "time_budget": TIME_BUDGET  # in minutes
    }
    return config


def results_dirname_generator():
    return f'unet++_efficientnet-b4_r_{MAX_ROUND}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}_fs_{FED_AGGREGATION_STRATEGY}' \
           f'_mf_{MIN_FIT_CLIENTS}_ff_{FRACTION_FIT}_do_{DICE_ONLY}_o_{OPTIMIZER_NAME}_lr_{LEARNING_RATE}_image_{IMAGE_SIZE}_IID'


def get_eval_fn(net):
    masks_path, images_path, labels = get_data_paths()
    test_dataset = LungSegDataset(path_to_images=images_path,
                                  path_to_masks=masks_path,
                                  image_size=IMAGE_SIZE,
                                  mode="test", labels=labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    def evaluate(weights):
        global ROUND, MAX_ROUND
        state_dict = get_state_dict(net, weights)
        net.load_state_dict(state_dict, strict=True)
        val_loss, val_jacc = validate(net, test_loader, DEVICE)
        res_dir = results_dirname_generator()
        if len(jacc) != 0 and val_jacc > max(jacc):
            unet_dir = os.path.join(res_dir, 'best_model')
            if os.path.exists(unet_dir):
                shutil.rmtree(unet_dir)
            os.mkdir(unet_dir)
            logger.info(f"Saving model as jaccard score is the best: {val_jacc}")
            torch.save(net.state_dict(), f'{unet_dir}/unet_{ROUND}_jacc_{round(val_jacc, 3)}_loss_{round(val_loss, 3)}')

        loss.append(val_loss)
        jacc.append(val_jacc)
        if MAX_ROUND == ROUND:
            df = pd.DataFrame.from_dict({'round': [i for i in range(MAX_ROUND + 1)], 'loss': loss, 'jaccard': jacc})
            df.to_csv(os.path.join(res_dir, 'result.csv'))
        ROUND += 1
        return val_loss, {"val_jacc": val_jacc, "val_dice_loss": val_loss}

    return evaluate


@click.command()
@click.option('--le', default=LOCAL_EPOCHS, type=int, help='Local epochs performed by clients')
@click.option('--a', default=FED_AGGREGATION_STRATEGY, type=str,
              help='Aggregation strategy (FedAvg, FedAdam, FedAdagrad')
@click.option('--c', default=CLIENTS, type=int, help='Clients number')
@click.option('--r', default=MAX_ROUND, type=int, help='Rounds of training')
@click.option('--mf', default=MIN_FIT_CLIENTS, type=int, help='Min fit clients')
@click.option('--ff', default=FRACTION_FIT, type=float, help='Fraction fit')
@click.option('--bs', default=BATCH_SIZE, type=int, help='Batch size')
@click.option('--lr', default=LEARNING_RATE, type=float, help='Learning rate')
@click.option('--o', default='Adam', type=str, help='Optimizer name (Adam, SGD, Adagrad')
def run_server(le, a, c, r, mf, ff, bs, lr, o):
    global OPTIMIZER_NAME, LOCAL_EPOCHS, FED_AGGREGATION_STRATEGY, CLIENTS, MAX_ROUND, MIN_FIT_CLIENTS, FRACTION_FIT, BATCH_SIZE, LEARNING_RATE
    LOCAL_EPOCHS = le
    FED_AGGREGATION_STRATEGY = a
    CLIENTS = c
    MAX_ROUND = r
    MIN_FIT_CLIENTS = mf
    FRACTION_FIT = ff
    BATCH_SIZE = bs
    LEARNING_RATE = lr
    OPTIMIZER_NAME = o

    logger.info(
        f"Configuration: le={LOCAL_EPOCHS}, clients={CLIENTS}, rounds={MAX_ROUND}, mf={MIN_FIT_CLIENTS}, ff={FRACTION_FIT}, bs={BATCH_SIZE}, lr={LEARNING_RATE}, opt={OPTIMIZER_NAME}")

    # Define model
    net = get_model().to(DEVICE)

    # Define strategy
    strategy = strategies[FED_AGGREGATION_STRATEGY](
        fraction_fit=FRACTION_FIT,
        fraction_eval=0.75,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_eval_clients=2,
        eval_fn=get_eval_fn(net),
        min_available_clients=CLIENTS,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.weights_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]),
    )

    # Start server
    server_addr = socket.gethostname()
    logger.info(f"Starting server on {server_addr}")

    res_dir = results_dirname_generator()
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.mkdir(res_dir)

    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": MAX_ROUND},
        strategy=strategy,
    )


if __name__ == "__main__":
    run_server()
