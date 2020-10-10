# mlflow-azure-recommender
mlflow example project for azure recommender algorithms

- This project runs mlflow experiments on SAR algorithm by using movielens100K data set.
- It saves the best model after hyperparameter tuning.
- Finally it serves the best model as a mlflow service.

- SAR algorithm is available at https://github.com/microsoft/recommenders

### Start Mlflow server to visualize experiments
```
mlflow server     --backend-store-uri sqlite:///mlflow.db     --default-artifact-root ./artifacts     --host 0.0.0.0
```

### Run training code
- train.py evaluates SAR algorithm by using different similarity metrics.
- Each experiment is saved to mlflow.
- Best model is saved as well.
- Remember to remove `sar_best` folder before running the script.
- Remember to make sure the dataset is available at `~/datasets/movielens100K/u.data` before running the script.
```
python train.py
```

```
Sar model (similarity_type= jaccard ):
  RMSE: 3.7950736953280937
Sar model (similarity_type= lift ):
  RMSE: 4.046395609396664
model_input    user_id
0        1
model output:    user_id  item_id  prediction
0        1      346    0.006996
1        1      844    0.006996
2        1      769    0.006996
3        1      394    0.006996
4        1      345    0.006996
5        1      849    0.006996
6        1      812    0.006996
7        1      795    0.006996
8        1      912    0.006996
9        1     1682    0.006996
```
- Remember to add `scipy` to `conda.yml` since we need it in the server, but it is not automatically added to the auto-generated file.
### Serve the saved best model
```
mlflow models serve -m sar_best -p 5001
```


### Test endpoint
```
curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{
    "data": {"user_id": [1]}
}'
```