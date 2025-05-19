import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eye_tracking_data/labels.csv")
plt.scatter(df["x"], df["y"], alpha=0.3, label="Original", c="blue")
plt.scatter(df["x"], 1 - df["y"], alpha=0.3, label="Y invertiert", c="red")
plt.scatter(df["y"], df["x"], alpha=0.3, label="X/Y vertauscht", c="green")
plt.xlabel("x")
plt.ylabel("y / Varianten")
plt.legend()
plt.grid(True)
plt.show()
