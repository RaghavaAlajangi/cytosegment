from torch.optim import lr_scheduler


def get_scheduler(config, optimizer):
    """Retrieves an optimizer instance based on the provided configuration."""
    scheduler_name = config.train.scheduler.name.lower()
    lr_decay_rate = config.train.scheduler.lr_decay_rate
    lr_step_size = config.train.scheduler.lr_step_size
    patience = config.train.scheduler.patience

    if scheduler_name == "steplr":
        return lr_scheduler.StepLR(
            optimizer=optimizer, step_size=lr_step_size, gamma=lr_decay_rate
        )

    if scheduler_name == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            patience=patience,
            factor=lr_decay_rate,
        )
