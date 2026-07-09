import random

from model import CNN
from fl_training import run_fl_rounds


def select_forgotten_samples(client_data, private_data):
    #Randomly select 1 sample per client to forget (paper Sec VI.B.1)
    forgotten = {}
    for client_id, sample_indices in client_data.items():
        forgotten[client_id] = random.choice(sample_indices)
    return forgotten


def retrain_without_samples(pretrained_sd, private_data, client_data,
                            forgotten_samples, round_selections):
    model = CNN()
    model.load_state_dict(pretrained_sd)
    modified_data = {}
    for client_id, sample_indices in client_data.items():
        if client_id in forgotten_samples:
            modified_data[client_id] = [i for i in sample_indices
                                        if i != forgotten_samples[client_id]]
        else:
            modified_data[client_id] = list(sample_indices)
    return run_fl_rounds(model, modified_data, private_data, round_selections)


#--- Forgotten-data sweep (paper Sec VII.A.2) ------------------------------

def build_forget_order(client_data):
    #Build, for every client, a fixed ORDER in which its samples are forgotten.
    #Forgetting the first N entries of this order (build_forget_sets below) gives
    #NESTED forget sets: the N=1 set is contained in the N=2 set, and so on. This
    #is the methodological anchor requested by the tutor: as N grows we ADD
    #samples to the forget set rather than swapping it, so a PSNR-vs-N curve
    #measures degradation "at parity of target image".
    #
    #The first entry of each client's order is chosen with the exact same
    #random.choice call sequence as select_forgotten_samples, so element 0 equals
    #the single sample that was forgotten (and reconstructed) in the N=1 baseline.
    #The remaining samples are shuffled once (deterministic under the global seed).
    firsts = {client_id: random.choice(sample_indices)
              for client_id, sample_indices in client_data.items()}

    order = {}
    for client_id, sample_indices in client_data.items():
        rest = [i for i in sample_indices if i != firsts[client_id]]
        random.shuffle(rest)
        order[client_id] = [firsts[client_id]] + rest
    return order


def build_forget_sets(forget_order, n_forget):
    #Take the first n_forget indices of each client's order -> the forget set for
    #this sweep point. Clipped so each client always keeps at least one sample.
    forget_sets = {}
    for client_id, ordered in forget_order.items():
        keep_at_least_one = min(n_forget, len(ordered) - 1)
        forget_sets[client_id] = list(ordered[:keep_at_least_one])
    return forget_sets


def retrain_without_sample_sets(pretrained_sd, private_data, client_data,
                                forget_sets, round_selections):
    #Retraining-based sample unlearning where each client removes a SET of
    #samples (forget_sets[client_id]) rather than a single one.
    model = CNN()
    model.load_state_dict(pretrained_sd)
    modified_data = {}
    for client_id, sample_indices in client_data.items():
        drop = set(forget_sets.get(client_id, []))
        modified_data[client_id] = [i for i in sample_indices if i not in drop]
    return run_fl_rounds(model, modified_data, private_data, round_selections)
