import pandas as pd
import matplotlib.pyplot as plt

# 1) CSVs einlesen
df_dqn = pd.read_csv("training_data_dqn_seed_0.csv")
df_rnd = pd.read_csv("training_data_seed_0_rnd.csv")

# 2) Einfaches Liniendiagramm
plt.figure(figsize=(8, 5))
plt.plot(df_dqn["steps"], df_dqn["rewards"], label="DQN")
plt.plot(df_rnd["steps"], df_rnd["rewards"], label="RND-DQN")

# 3) Achsenbeschriftungen und Titel
plt.xlabel("Environment Frames")
plt.ylabel("Episode Reward")
plt.title("DQN vs. RND-DQN auf CartPole-v1")

# 4) Legende und Raster
plt.legend()
plt.grid(True)

# 5) Plot anzeigen (oder speichern)
plt.tight_layout()
plt.show()
plt.savefig("comparison_plot.png", dpi=300)
