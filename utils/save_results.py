import pandas as pd

def save_results(errors,anomalies,path="results.csv"):
    df=pd.DataFrame({
        "error":errors,
        "anomaly":anomalies
    })
    df.to_csv(path,index=False)
    print(f"Results saved to {path}")