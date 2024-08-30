from torch.optim import Adam, SGD


def get_optimizer(config, model):
    """Retrieves an optimizer instance based on the provided configuration."""
    optimizer_name = config.train.optimizer.name.lower()
    learn_rate = config.train.optimizer.learn_rate

    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=learn_rate, eps=1e-07)

    if optimizer_name == "sgd":
        return SGD(model.parameters(), lr=learn_rate,
                   momentum=config.train.optimizer.momentum)
