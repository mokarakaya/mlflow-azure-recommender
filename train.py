import pandas as pd
import mlflow.sklearn
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import rmse
import sys
from sar_recommender_mlflow import SarRecommender



if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    dataset_path = '~/datasets/movielens100K/u.data'
    df = pd.read_csv(dataset_path, delimiter='\t')
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    train, test = python_stratified_split(df, ratio=0.75, col_user='user_id', col_item='item_id', seed=42)

    similarity_types = ['jaccard', 'lift']
    best_model = None
    best_score = sys.float_info.max
    for similarity_type in similarity_types:
        with mlflow.start_run():
            model = SarRecommender(similarity_type)
            model.fit(train)
            top_k = model.recommend_k_items(test, remove_seen=True)

            eval_rmse = rmse(test, top_k, col_user='user_id', col_item='item_id', col_rating='rating')
            if eval_rmse < best_score:
                best_score = eval_rmse
                best_model = model

            print("Sar model (similarity_type= {} ):".format(similarity_type))
            print("  RMSE: %s" % eval_rmse)
            mlflow.log_param("similarity_type", similarity_type)
            mlflow.log_metric("rmse", eval_rmse)

    model_path = 'sar_best'
    mlflow.pyfunc.save_model(path=model_path, python_model=best_model)

    # Load the model in `python_function` format
    loaded_model = mlflow.pyfunc.load_model(model_path)

    # Evaluate the model
    import pandas as pd

    model_input = pd.DataFrame({'user_id': [1]})
    model_output = loaded_model.predict(model_input)
    print('model output:', model_output)
