import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("kepler_exoplanet_data_cleaned.csv")
df=df.drop(columns=["koi_disposition"])

corr = df.select_dtypes(include='number').corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    cmap="vlag",
    annot=False,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75}
)
plt.title("Feature Correlation Heatmap – Kepler Exoplanet Data")
plt.tight_layout()
plt.show()