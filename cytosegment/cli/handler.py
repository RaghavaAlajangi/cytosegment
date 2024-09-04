import subprocess as sp
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ..training import Trainer

slurm_tmp = Path(__file__).parent / "templates" / "slurm_template.sh"


def get_experiment_path(experiment_config):
    """Returns the path to the experiment directory."""
    return (
        Path(experiment_config.run.dir)
        if Path(experiment_config.run.dir).is_dir()
        else Path(experiment_config.sweep.dir) / experiment_config.sweep.subdir
    )


def create_and_submit_slurm_job(experiment_path, experiment_config, config,
                                params_path):
    """Creates and submits a SLURM job for the specified experiment."""
    slurm_dir = experiment_path / "slurm_logs"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    slurm_path = experiment_path / "slurm_job.sh"

    user_overrides = HydraConfig.get().overrides.task
    # Run arguments
    kwargs = " ".join(user_overrides)

    job_dict = {
        "EXP_NAME": experiment_config.sweep.subdir,
        "PATH_OUT": config.path_out,
        "SLURM_LOGS": slurm_dir,
        "JOB_NAME": experiment_config.sweep.subdir,
        "MAIL_ID": config.hpc.mail_id,
        "MAX_MEM": int(config.hpc.max_mem_GB),
        "MAX_TIME": config.hpc.max_time_hours,
        "PARAMS_PATH": params_path,
        "KWARGS": kwargs
    }
    slurm_file = (slurm_tmp.read_text()).format(**job_dict)
    slurm_path.write_text(slurm_file)
    # to make it executable (needed for SLURM)
    sp.call("chmod +x " + str(slurm_path), shell=True)
    sp.check_output(f"sbatch {str(slurm_path)}", shell=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """
    This function serves as the entry point for the application, utilizing
    Hydra's main decorator to handle command line arguments and configuration.

    Parameters:
    -----------
    config: DictConfig
        The configuration object containing application settings.
    """

    experiment_config = HydraConfig.get()
    experiment_path = get_experiment_path(experiment_config)
    params_path = experiment_path / "run_params.yaml"
    # Override path_out with the experiment path
    OmegaConf.update(config, "path_out", str(experiment_path))
    # Save config as a yaml file
    OmegaConf.save(config, params_path)

    if config.slurm:
        create_and_submit_slurm_job(experiment_path, experiment_config, config,
                                    params_path)
    else:
        trainer = Trainer.with_params(config)
        trainer.start_train()
