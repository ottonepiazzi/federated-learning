import copy

from config import NUM_ROUNDS, FL_LR, LOCAL_EPOCHS, BATCH_SIZE, DEVICE
from model import CNN
from federated import client_update, fedavg, lr_schedule


#Retraining (deterministic): same initial state, same round schedule, target excluded
def retrain_without_client(init_sd, private_data, client_data,
                           target, round_selections):
    model = CNN()
    model.load_state_dict(init_sd)
    model.to(DEVICE)

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = [c for c in round_selections[rnd] if c != target]
        if not selected:
            continue
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)
        results = []
        for k in selected:
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, private_data, client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
        model = fedavg(model, results)

    return model
