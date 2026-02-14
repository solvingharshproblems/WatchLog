import matplotlib.pyplot as plt

def plot_anomalies(errors, anomalies):
    plt.figure(figsize=(12,5))
    plt.plot(errors, label="Reconstruction Error")
    plt.scatter(
        [i for i, a in enumerate(anomalies) if a],
        errors[anomalies],
        color="red",
        label="Anomaly"
    )
    plt.legend()
    plt.title("Log Anomaly Detection")
    plt.show()