import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)

import missingno as msno

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')

print('\n[ Shape ]')
print(df_train.shape)

print('\n\n[ Head ]')
print(df_train.head(5))

print('\n\n[ Describe ]')
print(df_train.describe())

print('\n\n[ Percent ]')
for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)

f, ax = plt.subplots(1, 2, figsize=(18, 9))
df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')
plt.show()