import surprise
import os

class Model():
    def __init__(self, hyper_params, user_count, item_count):
        latent_size = hyper_params['latent_size']

        if hyper_params['model_type'] == 'kNN':
            self.model = surprise.prediction_algorithms.knns.KNNBasic(k = 10, verbose = True)
        elif hyper_params['model_type'] == 'NMF':
            self.model = surprise.NMF(n_factors = latent_size, biased = False, n_epochs = 50, verbose = True)
        elif hyper_params['model_type'] == 'SVD':
            self.model = surprise.SVD(n_factors = latent_size, verbose = True)
        elif hyper_params['model_type'] == 'SVD++':
            self.model = surprise.SVDpp(n_factors = latent_size, verbose = True)
        elif hyper_params['model_type'] == 'baseline':
            bsl_options = {
                'method': 'sgd',
                'n_epochs': 20,
            }
            self.model = surprise.prediction_algorithms.baseline_only.BaselineOnly(bsl_options = bsl_options, verbose = True)

        self.hyper_params = hyper_params
        self.user_count = user_count
        self.item_count = item_count

    def __call__(self, rating_matrix, test_reader):
        print("Training..")
        self.model.fit(rating_matrix)

        print("Testing..")
        metrics, total = { 'MSE': 0.0 }, 0.0
        user_count_mse_map = {}
        item_count_mse_map = {}

        for (user, item), y in test_reader.iter_non_torch():

            for b in range(len(y)):
                predicted = self.model.predict(uid=str(user[b]), iid=str(item[b]))

                mse = (y[b] - predicted[3]) ** 2
                metrics['MSE'] += mse
                total += 1.0

                if user[b] not in self.user_count: self.user_count[user[b]] = 0
                if item[b] not in self.item_count: self.item_count[item[b]] = 0

                if self.user_count[user[b]] not in user_count_mse_map: user_count_mse_map[ self.user_count[user[b]] ] = []
                if self.item_count[item[b]] not in item_count_mse_map: item_count_mse_map[ self.item_count[item[b]] ] = []

                user_count_mse_map[ self.user_count[user[b]] ].append(mse)
                item_count_mse_map[ self.item_count[item[b]] ].append(mse)

        metrics['MSE'] /= total
        metrics['MSE'] = round(metrics['MSE'], 4)
        metrics['dataset'] = self.hyper_params['dataset']

        return metrics, user_count_mse_map, item_count_mse_map