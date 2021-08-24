import numpy as np

from tasksim import task_similarity

from graspologic.embed import AdjacencySpectralEmbed as ASE
from graspologic.cluster import AutoGMMCluster as GMM

from proglearn import LifelongClassificationForest as l2f

from graspologic.embed import ClassicalMDS as CMDS

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import pairwise_distances

def generate_hierarchical_gaussian_data(dist_means=None, n_clusts=2, n_dists_per_clust=2, d=2, clust_cov=1, n_per_dist=25, dist_cov=0.5, acorn=None):
    if n_clusts > 0:
        all_means = [
            np.array([1,1]),
            np.array([-1,-1]),
            np.array([1, -1]),
            np.array([-1, 1])
        ]
    else:
        all_means = [np.array([0,0])]
    
    clust_means = all_means[:n_clusts]
    clust_cov = clust_cov * np.eye(d)
    
    if dist_means is None:
        dist_means = [np.random.multivariate_normal(clust_means[i], clust_cov, size=n_dists_per_clust) for i in range(n_clusts)]
    else:
        n_clusts = len(dist_means)
        n_dists_per_clust, d = dist_means[0].shape
        
    dist_cov = dist_cov * np.eye(d)
    
    data = [
        [np.random.multivariate_normal(dist_means[i][j], dist_cov, size=n_per_dist) for j in range(n_dists_per_clust)] 
            for i in range(n_clusts)
    ]
            
    return dist_means, data

# def generate_dist_matrix(data, dissimilarity = 'task-sim', acorn=None):        
#     n_dists = len(data)    
#     labels = [i*np.ones(data[i].shape[0]) for i in range(n_dists)]
        
#     distances = np.zeros((n_dists, n_dists))
    
#     if dissimilarity == 'eucl-means':
#         means = np.mean(data, axis=1)
#         return pairwise_distances(means)
        
#     for i in range(n_dists):
#         for j in range(n_dists):
#             if i == j:
#                 continue
                
#             if dissimilarity == 'task-sim':
#                 for k in range(n_dists):
#                     if k == i or k == j:
#                         continue
                                                
#                     temp_task1 = (np.concatenate([data[i], data[k]], axis=0), np.concatenate([labels[i], labels[k]]))
#                     temp_task2 = (np.concatenate([data[j], data[k]], axis=0), np.concatenate([labels[j], labels[k]]))


#                     distances[i,j] += task_similarity(temp_task1, temp_task2)
#                 distances[i] /= n_dists-2
                
#             else:
#                 raise ValueError('other distances not implemented')
       
        
            
#     return distances

# def preprocess_dist_matrix(dist_matrix, make_symmetric=True, scale=True, aug_diag=True):
#     if make_symmetric:
#         dist_matrix = 0.5*(dist_matrix + dist_matrix.T)
        
#     if aug_diag:
#         dist_matrix = dist_matrix + np.diag(np.mean(dist_matrix, axis=0))
        
#     if scale:
#         dist_matrix = (dist_matrix - np.min(dist_matrix)) / (np.max(dist_matrix) - np.min(dist_matrix))
        
#     return dist_matrix

# def cluster_dists(dist_matrix, embedding=ASE, cluster=GMM):
#     if embedding is not None:
#         X_hat = embedding().fit_transform(dist_matrix)
#     else:
#         X_hat = dist_matrix
         
#     return cluster().fit_predict(X_hat)

# def evaluate_clusters(f, truth, preds, calculate_random=False, n_mc=500, acorn=None):
#     eval_pred = f(truth, preds)
    
#     if not calculate_random:
#         return eval_pred
    
#     eval_random = np.zeros(n_mc)
#     for i in range(n_mc):
#         np.random.shuffle(preds)
#         eval_random[i] = f(truth, preds)
        
#     return eval_pred, np.mean(eval_random)

# def evaluate_accuracy(data, labels, truth, preds, n_trees_coarse=25, n_trees_fine=10, train_flat=True,
#                      data_args = [],
#                      acorn=None):
#     forests_dict = {
#             'coarse_truth': None, 
#             'fine_truth': {c: None for c in np.unique(truth)},
#             'coarse_preds': None,
#             'fine_preds': {c: None for c in np.unique(preds)}, 
#             'flat': None
#     }
    
#     # Coarse forest
#     coarse_forest_truth = l2f(n_estimators=n_trees_coarse,
#                         default_finite_sample_correction=False,
#                         default_max_depth=20)
    
#     coarse_forest_truth.add_task(np.concatenate(data, axis=0), 
#                                  np.concatenate([truth[i] * np.ones(data[0].shape[0]) for i in range(len(truth))])
#                                 )
#     forests_dict['coarse_truth'] = coarse_forest_truth
    
#     coarse_forest_preds = l2f(n_estimators=n_trees_coarse,
#                         default_finite_sample_correction=False,
#                         default_max_depth=20)
    
#     coarse_forest_preds.add_task(np.concatenate(data, axis=0), 
#                                  np.concatenate([preds[i] * np.ones(data[0].shape[0]) for i in range(len(preds))])
#                                 )
#     forests_dict['coarse_preds'] = coarse_forest_preds
    
    
#     # Flat forest
#     n_trees_flat = n_trees_coarse + len(truth)*n_trees_fine
    
#     if train_flat:
#         flat_forest_truth = l2f(n_estimators=n_trees_flat,
#                             default_finite_sample_correction=False,
#                             default_max_depth=20)
#         flat_forest_truth.add_task(np.concatenate(data, axis=0), np.concatenate(labels))
#         forests_dict['flat'] = flat_forest_truth
        
#     # Fine forest
#     for j, parent_class in enumerate(np.unique(truth)):
#         temp_fine_indices = np.where(truth == parent_class)[0]
        
        
#         fine_forest_truth = l2f(n_estimators=n_trees_fine, 
#                                default_finite_sample_correction=False, 
#                                default_max_depth=20
#                               )
#         fine_forest_truth.add_task(np.concatenate(data[temp_fine_indices], axis=0), np.concatenate(labels[temp_fine_indices]))
#         forests_dict['fine_truth'][j] = fine_forest_truth
        
#     for j, parent_class in enumerate(np.unique(preds)):
#         temp_fine_indices = np.where(preds == parent_class)[0]
        
#         fine_forest_preds = l2f(n_estimators=n_trees_fine, 
#                                default_finite_sample_correction=False, 
#                                default_max_depth=20
#                               )
#         fine_forest_preds.add_task(np.concatenate(data[temp_fine_indices], axis=0), np.concatenate(labels[temp_fine_indices]))
#         forests_dict['fine_preds'][j] = fine_forest_preds
        
        
#     # Now, calculate accuracies
#     accuracies = np.zeros(3)
    
#     if data_args == []:
#         raise ValueError
        
#     n_dists = data_args[1] * data_args[2]
#     n_per_dist = data_args[5]
    
#     all_labels = np.concatenate(labels)
    
#     hierarchical_posteriors_truth = np.zeros((n_per_dist*n_dists, n_dists))
#     hierarchical_posteriors_preds = np.zeros((n_per_dist*n_dists, n_dists))
    
#     data_means, X_test = generate_hierarchical_gaussian_data(*data_args)
#     X_test = np.concatenate(X_test, axis=0)
#     labels_test = np.concatenate([i*np.ones(X_test[0].shape[0]) for i in range(n_dists)])

#     coarse_posteriors_truth = forests_dict['coarse_truth'].predict_proba(np.concatenate(X_test,axis=0), 0)
#     coarse_posteriors_preds = forests_dict['coarse_preds'].predict_proba(np.concatenate(X_test,axis=0), 0)
        
#     # Hierarchical posteriors & prediction
#     for j, parent_class in enumerate(np.unique(truth)):
#         temp_fine_label_indices = np.where(truth == parent_class)[0]
        
#         temp_fine_posteriors = forests_dict['fine_truth'][j].predict_proba(np.concatenate(X_test,axis=0), 0)
#         hierarchical_posteriors_truth[:, temp_fine_label_indices] = np.multiply(coarse_posteriors_truth[:, j],
#                                                                      temp_fine_posteriors.T
#                                                                     ).T
        
#     for j, parent_class in enumerate(np.unique(preds)):
#         temp_fine_label_indices = np.where(preds == parent_class)[0]

        
#         temp_fine_posteriors = forests_dict['fine_preds'][j].predict_proba(np.concatenate(X_test,axis=0), 0)
#         hierarchical_posteriors_preds[:, temp_fine_label_indices] = np.multiply(coarse_posteriors_preds[:, j],
#                                                                      temp_fine_posteriors.T
#                                                                     ).T
        
#     yhat_hc = np.argmax(hierarchical_posteriors_truth, axis=1)
#     accuracies[0] = np.mean(yhat_hc == np.array(labels_test))
    
#     yhat_hc = np.argmax(hierarchical_posteriors_preds, axis=1)
#     accuracies[1] = np.mean(yhat_hc == np.array(labels_test))
    
    
#     # Flat posteriors & prediction
#     if train_flat:
#         flat_posteriors = forests_dict['flat'].predict_proba(np.concatenate(X_test,axis=0), 0)
#         yhat_flat = np.argmax(flat_posteriors, axis=1)
#         accuracies[2] = np.mean(yhat_flat == np.array(labels_test))
    
#     return accuracies[:, np.newaxis].T
    
    
# def hierarchical_gaussian_exp(dist_means=None, n_clusts=2, n_dists_per_clust=2, d=2, clust_cov=1, n_per_dist=25, dist_cov=0.5,
#                               dissimilarity='task-sim',
#                               make_symmetric=True, scale=True, aug_diag=True,
#                               embedding=ASE, cluster=GMM,
#                               f=NMI, calculate_random=True, random_nmc=500,
#                               n_test_per_dist=250, n_trees_coarse=25, n_trees_fine=10, train_flat=True,
#                               acorn=None):
#     data_params = [dist_means, n_clusts, n_dists_per_clust, d, clust_cov, n_per_dist, dist_cov]
#     dist_params = dissimilarity
#     prep_params = (make_symmetric, scale, aug_diag)
#     cluster_params = (embedding, cluster)
    
#     means, data = generate_hierarchical_gaussian_data(*data_params)
#     dist_matrix = generate_dist_matrix(np.concatenate(data, axis=0), dist_params)
#     prep_dist_matrix = preprocess_dist_matrix(dist_matrix, *prep_params)
#     preds = cluster_dists(prep_dist_matrix, *cluster_params)
    
#     truth = [i*np.ones(n_dists_per_clust) for i in range(n_clusts)]
#     eval_params = (f, np.concatenate(truth), preds, calculate_random, random_nmc)
    
#     eval_pred, eval_random = evaluate_clusters(*eval_params)
    
#     data_params[0] = means
#     data_params[5] = n_test_per_dist
#     labels = np.array([i*np.ones(n_per_dist) for i in range(np.concatenate(data, axis=0).shape[0])])
    
#     acc_params = [np.concatenate(data, axis=0), labels, np.concatenate(truth), preds, n_trees_coarse, n_trees_fine, train_flat, data_params]
    
#     accs = evaluate_accuracy(*acc_params)
    
#     return np.array([eval_pred, eval_random])[:, np.newaxis].T, accs    
