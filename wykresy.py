import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Wczytanie danych
df = pd.read_csv("metrics.csv", sep=';')
print(df.head())
best_epoch = 27

sns.set(style="whitegrid", font_scale=1.2)

# ========================
# Loss - tren/val (osobny wykres)
# ========================
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linewidth=2)

plt.fill_between(df['epoch'], df['train_loss'], df['val_loss'], color='grey', alpha=0.2)

# Linia przerywana dla najlepszej epoki
plt.axvline(best_epoch, color='red', linestyle='--', linewidth=2, label='Best Epoch')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# ========================
# Metryki Audio (osobny wykres)
# ========================
plt.figure(figsize=(12, 6))
metrics = ['si_sdr', 'pesq', 'stoi', 'der']
colors = ['purple', 'green', 'orange', 'cyan']

for metric, color in zip(metrics, colors):
    plt.plot(df['epoch'], df[metric], label=metric.upper(), color=color, linewidth=2)

# Linia przerywana dla najlepszej epoki
plt.axvline(best_epoch, color='red', linestyle='--', linewidth=2, label='Best Epoch')

plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Audio Metrics Across Epochs')
plt.legend()
plt.grid(True)
plt.show()
