from reco_utils.recommender.sar import SAR
import mlflow.pyfunc

# Define the model class
class SarRecommender(mlflow.pyfunc.PythonModel):

    def __init__(self, similarity_type):
        self.model = SAR(
            col_user="user_id",
            col_item="item_id",
            col_rating="rating",
            col_timestamp="timestamp",
            similarity_type=similarity_type,
            time_decay_coefficient=30,
            timedecay_formula=True,
            normalize=True
        )

    def predict(self, context, model_input):
        print('context', context)
        print('model_input', model_input)
        recommendations = self.model.recommend_k_items(model_input, remove_seen=True)
        print('recommendations for users {} are {}'.format(model_input, recommendations))
        return recommendations

    def recommend_k_items(self, test, top_k=10, sort_top_k=True, remove_seen=True):
        return self.model.recommend_k_items(test, top_k=top_k, sort_top_k=sort_top_k, remove_seen=remove_seen)

    def fit(self, df):
        return self.model.fit(df)