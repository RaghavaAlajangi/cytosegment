class EarlyStoppingCallback:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.last_loss = None

    def step(self, current_loss):
        if self.last_loss is None:
            self.last_loss = current_loss
        if self.last_loss < current_loss:
            self.counter += 1
        else:
            self.last_loss > current_loss
            self.counter = 0

    def should_stop(self):
        if self.counter > self.patience:
            return True
        else:
            return False
