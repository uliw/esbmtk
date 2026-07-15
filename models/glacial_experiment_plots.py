import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("results_summary.csv")

# Set seaborn style for a professional look
sns.set(style="whitegrid", palette="muted")

# ------------------- Bar plot: final atmospheric CO2 -------------------
plt.figure(figsize=(10,6))

# Create barplot
ax = sns.barplot(x="experiment", y="CO2_ppm", data=df)

# Rotate x-axis labels slightly and center them
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)

# Set y-axis label and limits
ax.set_ylabel("Atmospheric CO₂ (ppm)", fontsize=12)
ax.set_ylim(180, 300)

# Add title
ax.set_title("Final Atmospheric CO₂ Across Experiments", fontsize=14, weight='bold')

# Annotate each bar with the CO2 value
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}',  # show one decimal
                xy=(p.get_x() + p.get_width() / 2, height),  # center top
                xytext=(0, 5),  # offset above bar
                textcoords='offset points',
                ha='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()



# ------------------- Grouped Bar Plot: Carbonate Saturation (zcc) -------------------
zcc_cols = ["A_zcc", "I_zcc", "P_zcc"]
zcc_df = df.melt(id_vars="experiment", value_vars=zcc_cols,
                 var_name="Ocean", value_name="zcc")

# Map column names to prettier labels
ocean_labels = {"A_zcc": "Atlantic", "I_zcc": "Indian", "P_zcc": "Pacific"}
zcc_df["Ocean"] = zcc_df["Ocean"].map(ocean_labels)

plt.figure(figsize=(14,6))
ax = sns.barplot(x="experiment", y="zcc", hue="Ocean", data=zcc_df)

# Center x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)

# Y-axis label
ax.set_ylabel("Carbonate Compensation Depth (m)", fontsize=12)

# Title
ax.set_title("Final Carbonate Compensation Depth (m) Across Experiments", fontsize=14, weight='bold')

# Annotate bars with values
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.0f}',  # show two decimals
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', fontsize=9, weight='bold')

# Adjust legend
ax.legend(title="Ocean", fontsize=10, title_fontsize=11)

plt.tight_layout()
plt.show()



# Melt for Atlantic/Indian/Pacific zcc
zcc_cols = ["A_zcc", "I_zcc", "P_zcc"]
zcc_df = df.melt(id_vars="experiment", value_vars=zcc_cols,
                 var_name="Ocean", value_name="zcc")

# Map ocean labels
ocean_labels = {"A_zcc": "Atlantic", "I_zcc": "Indian", "P_zcc": "Pacific"}
zcc_df["Ocean"] = zcc_df["Ocean"].map(ocean_labels)

plt.figure(figsize=(14,6))
ax = sns.barplot(x="experiment", y="zcc", hue="Ocean", data=zcc_df)

# Flip y-axis so depth increases downward
ax.invert_yaxis()

# Move x-axis to top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# Center x-axis labels and adjust font
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)

# Labels and title
ax.set_ylabel("Carbonate Compensation Depth (m)", fontsize=12)
ax.set_xlabel("Experiment", fontsize=12)
ax.set_title("Final Carbonate Compensation Depth Across Experiments", fontsize=14, weight='bold', pad=20)

# Annotate bars with values below each bar
for p in ax.patches:
    height = p.get_height()
    # Since y-axis is inverted, place text slightly above the top of the bar
    ax.annotate(f'{height:.0f}',
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, -12),
                textcoords='offset points',
                ha='center', fontsize=9, weight='bold', color='black')

ax.legend(title="Ocean", fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()



# Load the timeseries CSV
df = pd.read_csv("results_timeseries.csv")

# --------------------- CO2 Time Series ---------------------
plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.lineplot(data=df, x="time", y="CO2_ppm", hue="experiment", palette="tab10")
plt.xlabel("Time (kyr)")
plt.ylabel("Atmospheric CO2 (ppm)")
plt.title("Atmospheric CO2 Over Time Across Experiments")
plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim(df["time"].min(), df["time"].max())
plt.ylim(190, 300)  # optional if you want to focus on realistic CO2 range
plt.tight_layout()
plt.show()
