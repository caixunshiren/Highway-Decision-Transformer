def train(model, dataloader, config):
    device = config['device']
    model.to(device)

    # dimensions of the state dimension and the action dimension
    state_dim = config['state_dim']
    act_dim = config['act_dim']
