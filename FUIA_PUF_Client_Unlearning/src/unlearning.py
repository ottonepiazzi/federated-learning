import copy
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset

from config import NUM_ROUNDS, FL_LR, LOCAL_EPOCHS, BATCH_SIZE, DEVICE
from model import CNN
from federated import client_update, fedavg, lr_schedule, evaluate


#Retraining (deterministic): replays same round selections minus target
def retrain_without_client(pretrained_sd, private_data, client_data,
                           target, round_selections):
    model = CNN()
    model.load_state_dict(pretrained_sd)

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


#PUF-Special unlearning (paper Algorithm 1 with S_t^+ = empty, Eq. 9)
def puf_special_unlearn(global_model, private_dataset, client_data_indices,
                        target_client, eta_u,
                        local_epochs, batch_size, learning_rate, device):
    global_state_dict = copy.deepcopy(global_model.state_dict())

    #Target client performs one round of regular local training (ClientOpt).
    target_local_model = copy.deepcopy(global_model)
    target_state_dict, _, _ = client_update(
        target_local_model,
        private_dataset,
        client_data_indices[target_client],
        local_epochs,
        batch_size,
        learning_rate,
        device,
    )

    #Pseudo-gradient (target client's model update on w_t).
    pseudo_gradient = {
        parameter_name: target_state_dict[parameter_name].float()
                        - global_state_dict[parameter_name].float()
        for parameter_name in target_state_dict
    }

    #Apply scaled negation: w_unlearned = w_t - eta_u * pseudo_gradient
    unlearned_state_dict = OrderedDict()
    for parameter_name in global_state_dict:
        unlearned_state_dict[parameter_name] = (
            global_state_dict[parameter_name].float()
            - eta_u * pseudo_gradient[parameter_name]
        )

    unlearned_model = CNN()
    unlearned_model.load_state_dict(unlearned_state_dict)
    return unlearned_model


#Forget Accuracy: accuracy on the target client's data (the data being forgotten)
def forget_accuracy(model, private_dataset, target_indices, device):
    forget_loader = DataLoader(
        Subset(private_dataset, target_indices),
        batch_size=64,
        shuffle=False,
    )
    return evaluate(model, forget_loader, device)
