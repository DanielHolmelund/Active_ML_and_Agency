import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

file_name = 'data_237.csv'
file_name1 = 'data_318.csv'
df = pd.read_csv(file_name)
df = df[['Q','R','S','T']]
df_50 = df.sample(frac=0.5)

Q = df['Q']
R = df['R']
S = df['S']
T = df['T']

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1,8)

ax1.hist(df['Q'])
ax2.hist(df['R'])
ax3.hist(df['S'])
ax4.hist(df['T'])
ax5.hist(df_50['Q'])
ax6.hist(df_50['R'])
ax7.hist(df_50['S'])
ax8.hist(df_50['T'])
plt.show()

print('Q)','100 samples mean: {},'.format(df['Q'].mean()),'100 samples var: {},'.format(df['Q'].var()),'50 samples mean: {},'.format(df_50['Q'].mean()),'50 samples var: {}'.format(df_50['Q'].var()))
print('R)','100 samples mean: {},'.format(df['R'].mean()),'100 samples var: {},'.format(df['R'].var()),'50 samples mean: {},'.format(df_50['R'].mean()),'50 samples var: {}'.format(df_50['R'].var()))
print('S)','100 samples mean: {},'.format(df['S'].mean()),'100 samples var: {},'.format(df['S'].var()),'50 samples mean: {},'.format(df_50['S'].mean()),'50 samples var: {}'.format(df_50['S'].var()))
print('T)','100 samples mean: {},'.format(df['T'].mean()),'100 samples var: {},'.format(df['T'].var()),'50 samples mean: {},'.format(df_50['T'].mean()),'50 samples var: {}'.format(df_50['T'].var()))
print('100 samples: ')
print(df.corr())
print('50 samples: ')
print(df_50.corr())


'''

#Lets Something
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(df['S']+df['R'])
ax2.hist(df['T'])
plt.title('T=S+Q')
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(df['Q']+df['R']+df['T'])
ax2.hist(df['S'])
plt.title('S=Q+R+T')
plt.show()

#Interveening on q=1:

file_name = 'data_273.csv'
df_int = pd.read_csv(file_name)
df_int.drop([0])

Q_int = df_int['Q']
R_int = df_int['R']
S_int = df_int['S']
T_int = df_int['T']

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1,8)
ax1.hist(df['Q'], bins=20)
ax2.hist(df['R'], bins=20)
ax3.hist(df['S'], bins=20)
ax4.hist(df['T'], bins=20)
ax5.hist(Q_int, bins=20)
ax6.hist(R_int, bins=20)
ax7.hist(S_int, bins=20)
ax8.hist(T_int, bins=20)
plt.title('Q=1')
plt.show()

file_name = 'data_237.csv'
df = pd.read_csv(file_name)
df.drop([0])

file_name = 'data_273.csv'
df_int_H = pd.read_csv(file_name)
df_int_H.drop([0])

file_name = 'data_246.csv'
df_int_S = pd.read_csv(file_name)
df_int_S.drop([0])

file_name = 'data_258.csv'
df_int_q = pd.read_csv(file_name)
df_int_q.drop([0])

file_name = 'data_267.csv'
df_int_t = pd.read_csv(file_name)
df_int_t.drop([0])

file_name = 'data_269.csv'
df_int_r = pd.read_csv(file_name)
df_int_t.drop([0])



print(df.describe())
print(df_int_H.describe())
print(df_int_S.describe())
print(df_int_q.describe())
print(df_int_r.describe())
print(df_int_t.describe())

'''