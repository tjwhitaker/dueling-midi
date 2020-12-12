import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

init_times = pd.read_csv("time_model_init.csv", header=None).transpose()
training_times = pd.read_csv("time_training.csv", header=None).transpose()
inference_times = pd.read_csv("time_inference.csv", header=None).transpose()

init_times.columns = init_times.iloc[0]
init_times.drop(0, inplace=True)
init_times = init_times.astype(float)

training_times.columns = training_times.iloc[0]
training_times.drop(0, inplace=True)
training_times = training_times.astype(float)

inference_times.columns = inference_times.iloc[0]
inference_times.drop(0, inplace=True)
inference_times = inference_times.astype(float)

print(init_times.describe())
print(training_times.describe())
print(inference_times.describe())

# init_times.plot.box()
# plt.title("Initialization Times")
# plt.ylabel("Time (s)")
# plt.show()

# training_times.plot.box()
# plt.title("Training Times per Epoch")
# plt.ylabel("Time (s)")
# plt.show()

# inference_times.plot.box()
# plt.title("Inference Times for 32 Note Sequence")
# plt.ylabel("Time (s)")
# plt.show()
