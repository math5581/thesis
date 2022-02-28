from statsmodels.stats.weightstats import ztest as ztest
from utilities.utilities import *
import scipy
import numpy as np
import matplotlib.pyplot as plt


cos_vec_gt_224 = np.asarray(load_similarity_vector('/workspace/similarity_data/cosine/MOT17_02/sim_vec_gt.pkl'))
cos_vec_avg_not_gt_224 = np.asarray(load_similarity_vector('/workspace/similarity_data/cosine/MOT17_02/sim_vec_avg_not_gt.pkl'))
cos_vec_max_not_gt_224 = np.asarray(load_similarity_vector('/workspace/similarity_data/cosine/MOT17_02/sim_vec_max_not_gt.pkl'))





euc_vec_gt_224 = np.asarray(load_similarity_vector('/workspace/similarity_data/eucledian_dist/MOT17_02/sim_vec_224_gt.pkl'))
euc_vec_avg_not_gt_224 = np.asarray(load_similarity_vector('/workspace/similarity_data/eucledian_dist/MOT17_02/sim_vec_avg_not_gt_224.pkl'))
euc_vec_max_not_gt_224 = np.asarray(load_similarity_vector('/workspace/similarity_data/eucledian_dist/MOT17_02/sim_vec_max_not_gt_224.pkl'))

#print('224, euc 1 and 2 ', ztest(euc_vec_gt_224, euc_vec_max_not_gt_224))
#print('224, euc 1 and 3 ', ztest(euc_vec_gt_224, euc_vec_avg_not_gt_224))

test = scipy.stats.kruskal(euc_vec_gt_224, euc_vec_max_not_gt_224)

print(test)

# plt.hist(euc_vec_gt_224, bins = bin)
# plt.savefig('test_fig.png')
