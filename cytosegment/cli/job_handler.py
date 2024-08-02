import subprocess as sp
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ..ml_trainer import SetupTrainer

slurm_tmp = Path(__file__).parent / "templates" / "slurm_template.sh"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    hyconf = HydraConfig.get()

    # Get the experiment path
    run_path = Path(hyconf.run.dir) if Path(hyconf.run.dir).is_dir() else Path(
        hyconf.sweep.dir) / hyconf.sweep.subdir

    run_params_path = run_path / "run_params.yaml"

    OmegaConf.save(config, run_params_path)

    if config.slurm:
        # Read the slurm job template
        slurm_file = slurm_tmp.read_text()
        # Create directory to save hpc output files
        slurm_dir = run_path / "slurm_files"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        # Create experiment params and slurm job files
        slurm_path = run_path / "slurm_job.sh"
        hpc_job_dict = {
            "EXP_NAME": hyconf.sweep.subdir,
            "PATH_OUT": config.path_out,
            "JOB_NAME": hyconf.sweep.subdir,
            "MAIL_ID": config.cluster.mail_id,
            "MAX_MEM": int(config.cluster.max_mem_GB),
            "MAX_TIME": config.cluster.max_time_hours,
            "PARAMS_PATH": run_params_path
        }
        # Update and save slurm job file
        for k, v in hpc_job_dict.items():
            slurm_file = slurm_file.replace("{{" + k + "}}", str(v))
        slurm_path.write_text(slurm_file)
        # Run slurm jobs
        slout = sp.check_output(f"sbatch {str(slurm_path)}", shell=True)

        for line in slout.decode().split("\n"):
            line = line.strip().lower()
            if line.startswith("submitted batch job"):
                print(line)
    else:
        trainer = SetupTrainer.with_params(config)
        trainer.start_train()
