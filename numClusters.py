from clusteringMethods import ClusteringMethods
import math
import numpy as np

class NumClusters:
    """ Estimates the number of clusters a data set should have.
        :param df :type pd.DataFrame: dataframe with token columns
        suggested_bounds :type tuple[int, int]: an estimate of num_clusters to shorten computation time
        NOTE: suggested_bounds only applies for non-processed data with plain vectorization  """
    def __init__(self, df):
        self.df = df
        lower_bound = int((df.shape[0] * df.shape[1]) / np.count_nonzero(df))
        upper_bound = min(int((df.shape[1] / 2) ** 0.5), df.shape[0])
        self.suggested_bounds = (lower_bound, upper_bound)
        self.BEST_K, self.BEST_SCORE, self.CALCULATED_NUMS = 0, 0.0, {}

    def __get_score(self, k: int, cluster_model) -> float:
        if k in self.CALCULATED_NUMS:
            return self.CALCULATED_NUMS[k]
        else:
            from sklearn.metrics import silhouette_score
            pred = ClusteringMethods(self.df, cluster_model=cluster_model, n_clusters=k).pred_labels
            score = silhouette_score(self.df, pred)
            self.CALCULATED_NUMS[k] = score
            return score

    def silhouette(self, bounds=None, margin=0.05, cluster_model='agglomerative', print_progress=False) -> int:
        """ :param bounds :type Optional (int, int) : bounds for possible num_clust.
                If None, uses largest possible bounds
            :param margin :type (float, float) w/ num between 0 and 1: algorithm continues so long as score is
                within (1 - margin) * BEST_SCORE and BEST_SCORE
            :param cluster_model: (see ClusteringMethods.model_dict)
            :param print_progress :type bool: if True prints progress statements during computation
            :return BEST_K :type int: number of clusters estimation  """
        # if bounds is None, set default bounds
        bounds = (2 + math.ceil((self.df.shape[0] - 2) / 5), self.df.shape[0]) if not bounds else bounds
        interval = math.ceil((bounds[1] - bounds[0]) / 5)  # partition into 5 sections
        if interval <= 5:
            # return math.ceil(mean([self.BEST_K, bounds[1]]))
            return self.BEST_K
        print(f'Scoring {bounds[0]} to {bounds[1]} with interval {interval}...') if print_progress else None
        for k in range(bounds[0], bounds[1], interval):
            score = self.__get_score(k, cluster_model)
            print(k, score, self.BEST_K, self.BEST_SCORE) if print_progress else None
            if score >= self.BEST_SCORE:  # if score is increasing
                self.BEST_SCORE, self.BEST_K = score, k  # replace best score and k
            elif self.BEST_SCORE * (1 - margin) <= score < self.BEST_SCORE:  # if score is decreasing but within margin
                self.BEST_K = k if k > self.BEST_K else self.BEST_K  # replace best k and continue
            elif score < self.BEST_SCORE * (1 - margin) and self.BEST_K < k:  # if score is decreasing outside margin
                bounds = (self.BEST_K - interval, self.BEST_K + interval)
                return self.silhouette(bounds, margin, cluster_model, print_progress)  # start next loop

            if k == range(bounds[0], bounds[1], interval)[-1]:  # at the end of loop
                bounds = (self.BEST_K - interval, self.BEST_K + interval)
                return self.silhouette(bounds, margin, cluster_model, print_progress)  # start next loop
        print('Number of Clusters', self.BEST_K) if print_progress else None
        return self.BEST_K