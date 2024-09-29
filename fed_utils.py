import torch
import copy
from prettytable import PrettyTable
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

def average_weights(w,idxs_users,datanumber_client,islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    
    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points
        
        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg

def cluster_weights(w, datanumber):
    propmt_cluster = []
    for i in range(len(w)):
        prompt = w[i]['prompt_learner.ctx'].flatten(0).cpu()
        propmt_cluster.append(prompt.numpy())

    # cluster_matrix = linkage(propmt_cluster, 'average')
    # cluster_results = fcluster(cluster_matrix, 1, 'distance')
    cluster_model = AgglomerativeClustering(n_clusters=3, linkage="average", affinity="cosine")
    cluster_model = cluster_model.fit(propmt_cluster)
    cluster_results = cluster_model.labels_
    cluster_number = max(cluster_results) + 1
    cluster_group = [[] for i in range(cluster_number)]
    w_cluster = {cluster_i: None for cluster_i in range(cluster_number)}
    w_temp = copy.deepcopy(w[0])

    for idx in range(len(cluster_results)):
        cluster_group[cluster_results[idx]].append(idx)

    for num in range(cluster_number):
        client_list = cluster_group[num]
        total_data_points = sum([datanumber[r] for r in client_list])
        fed_avg_freqs = [datanumber[r] / total_data_points for r in client_list]
        for idx in range(len(client_list)):
            if idx == 0:
                prompt_avg = w[client_list[idx]]['prompt_learner.ctx'] * fed_avg_freqs[idx]
            else:
                prompt_avg += w[client_list[idx]]['prompt_learner.ctx'] * fed_avg_freqs[idx]
        w_temp['prompt_learner.ctx'] = prompt_avg
        w_cluster[num] = w_temp

    return w_cluster, cluster_group

def count_parameters(model,model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params