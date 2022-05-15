import warnings

import ray
from flaml import CFO, BlendSearch
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import AUC_at_k, ndcg_at_k, train_test_split
from ray import tune

NUM_THREADS = 8

ray.init(address="auto")


def train_model(config, data=None):
    train, test = train_test_split(
        data, train_percentage=0.9, random_state=42
    )
    bpr = BayesianPersonalizedRanking(
        factors=8,
        regularization=config["regularization"],
        learning_rate=config["learning_rate"],
        iterations=config["iterations"],
        verify_negative_samples=True,
        num_threads=NUM_THREADS,
    )
    bpr.fit(train, show_progress=False)
    auc = AUC_at_k(bpr, train, test, K=10, num_threads=NUM_THREADS)
    ndcg = ndcg_at_k(bpr, train, test, K=3, num_threads=NUM_THREADS)
    tune.report(auc=auc, ndcg=ndcg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # NB don't forget to clear ~/ray_results/ and ~/implicit_datasets/

    data = get_movielens("10m")
    _, arr = data
    data = arr.tocoo()

    config = {
        "regularization": tune.loguniform(0.000001, 0.1),
        "learning_rate": tune.loguniform(0.000001, 0.1),
        "iterations": tune.lograndint(50, 500),
    }

    low = {
        "regularization": 0.003,
        "learning_rate": 0.0025,
        "iterations": 50,
    }

    target_metric = "auc"
    search_alg = BlendSearch(
        # search_alg = CFO(
        metric=target_metric,
        mode="max",
        low_cost_partial_config=low,
    )

    analysis = tune.run(
        tune.with_parameters(train_model, data=data),
        config=config,
        metric=target_metric,
        mode="max",
        num_samples=500,
        keep_checkpoints_num=10,
        checkpoint_at_end=True,
        raise_on_failed_trial=False,
        verbose=1,
        time_budget_s=3600,  # 1 hour
        search_alg=search_alg,
    )

    df = analysis.results_df.sort_values(
        by=target_metric,
        axis=0,
        ascending=False,
    )
    print("\n\n\n")
    print(df, flush=True)

    print(
        "\n\n\nBest config: ",
        analysis.get_best_config(metric="geo", mode="max"),
    )
