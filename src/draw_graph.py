import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
import json

import pandas as pd
filepath = os.path.join(DATA_DIRECTORY, 'pan_error', '04_015_01.json')
image_filepath = os.path.join(DATA_DIRECTORY, 'pan_error', '04_015_01.png')

with open(filepath) as fp:
    data = json.load(fp)

df = pd.DataFrame(data)
df = df.drop_duplicates()

plt.plot(df['t'], df['error'], marker='')
plt.title(image_filepath)
plt.savefig(image_filepath)

# plt.show()
# # print(data)