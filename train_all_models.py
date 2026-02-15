from model.preprocessing import load_data, preprocess_data, split_data
from model.train_models import build_models
from model.evaluate import evaluate_model
from model.utils import save_model
import pandas as pd

DATA_PATH = "/content/sample_data/bank-marketing-ml/data/bank.csv"

df = load_data(DATA_PATH)
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

pipelines = build_models(preprocessor)

results = []

for name, model in pipelines.items():
    model.fit(X_train, y_train)
    metrics, cm = evaluate_model(model, X_test, y_test)
    save_model(model, f"{name.replace(' ', '_').lower()}.pkl")

    metrics["Model"] = name
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/metrics_summary.csv", index=False)

print(results_df)