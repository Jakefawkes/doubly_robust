
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.spatial.distance import pdist
from policy_evaluation.Estimator import Estimator
import scipy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from policy_evaluation.Policy import *
from scipy.stats.mstats import winsorize

class propensity_estimate_cpu():
    def __init__(self,X,y):
        # self.classifier = LogisticRegressionCV(cv=10)
        self.classifier = MLPClassifier(hidden_layer_sizes=(200,200),
                                        activation='tanh',
                                        learning_rate_init=0.01,
                                        early_stopping=True,
                                        batch_size=X.shape[0],
                                        )
        self.classifier.fit(X,y)

    def predict_score(self,X_test):
        return self.classifier.predict_proba(X_test)[:,1]

class DRCFMEstimator(Estimator):
    @property
    def name(self):
        return "dr_cmfe_estimator"

    def __init__(self, context_kernel, recom_kernel,null_policy: MultinomialPolicy, target_policy: MultinomialPolicy, params):
        """
         :param context_kernel: the kernel function for the context variable
         :param recom_kernel: the kernel function for the recommendation
         :param params: all parameters including regularization parameter and kernel parameters
         """

        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self.params = params
        self.null_policy = null_policy
        self.target_policy = target_policy
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def calculate_weight(self, row):
        nullProb = self.null_policy.get_propensity(row.null_multinomial, row.null_reco)
        if not self.target_policy.greedy:
            targetProb = self.target_policy.get_propensity(row.target_multinomial, row.null_reco)
        else:
            targetProb = 0.0 if row.null_reco == row.target_reco else -1e99

        return np.exp(targetProb - nullProb)

    def estimate(self, sim_data):
        """
         Calculate and return a coefficient vector (beta) of the counterfactual mean embedding of reward distribution.
         """

        # extract the regularization and kernel parameters
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]

        null_reward = sim_data.null_reward.dropna(axis=0)

        null_context_vec = np.stack(sim_data['null_context_vec'].dropna(axis=0).values)
        null_reco_vec = np.stack(sim_data['null_reco_vec'].dropna(axis=0).values)
        target_context_vec = np.stack(sim_data['target_context_vec'].dropna(axis=0).values)
        target_reco_vec = np.stack(sim_data['target_reco_vec'].dropna(axis=0).values)

        # use median heuristic for the bandwidth parameters
        context_param = (0.5 * context_param) / np.median(pdist(np.vstack([null_context_vec, target_context_vec]), 'sqeuclidean'))
        recom_param = (0.5 * recom_param) / np.median(pdist(np.vstack([null_reco_vec, target_reco_vec]), 'sqeuclidean'))

        contextMatrix = self.context_kernel(null_context_vec, null_context_vec, context_param)
        recomMatrix = self.recom_kernel(null_reco_vec, null_reco_vec, recom_param)  #
        A = contextMatrix*recomMatrix

        targetContextMatrix = self.context_kernel(null_context_vec, target_context_vec, context_param)
        targetRecomMatrix = self.recom_kernel(null_reco_vec, target_reco_vec, recom_param)
        b = contextMatrix*targetRecomMatrix
        # calculate the coefficient vector using the pointwise product kernel L_ij = K_ij.G_ij
        m = sim_data["target_reco"].dropna(axis=0).shape[0]
        n = sim_data["null_reco"].dropna(axis=0).shape[0]

        # prop_labels = np.concatenate([np.ones(m),np.zeros(n)],axis=0)
        #
        # target_cov_data = np.concatenate([target_context_vec,target_reco_vec],axis=1)
        # null_cov_data = np.concatenate([null_context_vec,null_reco_vec],axis=1)
        # prop_data = np.concatenate([target_cov_data,null_cov_data
        #                             ],axis=0)
        # c = propensity_estimate_cpu(X=prop_data,y=prop_labels)
        # e = c.predict_score(target_cov_data)
        # e_1 = e

        # ips_w = winsorize(sim_data.apply(self.calculate_weight, axis=1), (0.0, 0.01))
        ips_w = winsorize(sim_data.apply(self.calculate_weight, axis=1), (0.0, 0.005))
        e_1 = np.array(ips_w)#.clip(min=0,max=150)


        # ips_w =sim_data.apply(self.calculate_weight, axis=1)#, (0.0, 0.01))
        # e_1 = np.array(ips_w).clip(min=0,max=150)
        # e_1 = np.array(ips_w)

        b_target = b#.sum(axis=1)
        b_null = A#.sum(axis=1)#(A@e_1)/n

        #TODO weighting here is weird something is up!
        # b = b#@((1-e_0)/e_0)/m
        # solve a linear least-square
        A = A + np.diag(np.repeat(n * reg_param, n))
        r_inv, _ = scipy.sparse.linalg.cg(A, null_reward, tol=1e-06)
        # beta_null, _ = scipy.sparse.linalg.cg(A, b_null, tol=1e-06)
        # r_1 = np.dot(beta_target,null_reward)#/beta_target.sum()
        # res = null_reward- np.dot(beta_null,null_reward)#/beta_null.sum()
        # r_2 = np.mean(e_1*res)
        # middle_term = np.mean(e_1*null_reward)#/e_1.sum()
        # r_2 = np.dot(beta_null,null_reward)#/beta_null.sum()
        # dr_exp = r_1+r_2
        r_1 = r_inv @ b_target
        r_2 = r_inv @ b_null
        dr_exp = r_1 + e_1 * (null_reward-r_2)

        dr_exp = dr_exp.mean()
        return dr_exp #/ r_inv.sum()
        # return the expected reward as an average of the rewards, obtained from the null policy,
        # weighted by the coefficients beta from the counterfactual mean estimator.
        # return
        # return # np.dot(beta_vec, null_reward) -

