import time
from datetime import timedelta
from pathlib import Path

import click

from unet.ml_trainer import SetTrainer


@click.command(help="Script to run the training")
@click.option("--params_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to params file (.yaml)")
def main(params_path):

    trainer = SetTrainer.with_params(params_path)

    tik = time.time()
    print("Started training.....")
    trainer.train()
    tok = time.time() - tik
    train_time = str(timedelta(seconds=tok)).split('.')[0]
    print(f"Total training time: {train_time}")


if __name__ == "__main__":
    main()
