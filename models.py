import joblib

# Suppose you have these trained pipelines: pipe_price, pipe_area, pipe_age,
# pipe_status, pipe_region, pipe_locality
all_models = {
    "price": pipe_price,
    "area": pipe_area,
    "age": pipe_age,            # numeric regression or classifier (choose one)
    "status": pipe_status,
    "region": pipe_region,
    "locality": pipe_locality
}
joblib.dump(all_models, "model_artifacts/multi_model.joblib")
