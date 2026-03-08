from ast import Import

import joblib

artifact = joblib.load("model.pkl")

print(type(artifact))
print(artifact.keys())

print("\nmodels keys:", artifact["models"].keys())
print("feature_columns count:", len(artifact["feature_columns"]))
print("metrics:", artifact["metrics"])
