import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bike_component_data.csv")

# Visual 1: Pairplot for correlations
sns.pairplot(df, hue="will_fail_soon")
plt.suptitle("Feature Relationships with Failure", y=1.02)
plt.show()

# Visual 2: Feature correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Visual 3: Average failure rate by road roughness
plt.figure(figsize=(7, 4))
sns.barplot(x="road_roughness_index", y="will_fail_soon", hue="road_roughness_index", 
            data=df, palette="mako", legend=False)
plt.title("Failure Probability vs. Road Roughness Index")
plt.show()
