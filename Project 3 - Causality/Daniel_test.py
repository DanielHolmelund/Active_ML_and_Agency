import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

file_name = "data_211_K=0.csv"
#data = np.genfromtxt("data_85.csv", delimiter = ",")
#print(data)

Tot = 6 # Number of subplots
cols = 3 # Number of cols

rows = Tot // cols
rows += Tot % cols
position = range(1, Tot + 1)
fig = plt.figure(1)

df = pd.read_csv(file_name)
df = df.drop(df.columns[0], axis = 1)


k = 0
name = df.columns
for column in df:
    ax = fig.add_subplot(rows, cols, position[k])
    _, bins, _ = plt.hist(df[column], bins = 15, density=1, alpha=0.5)
    mu, sigma = scipy.stats.norm.fit(df[column])
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.title("{name}, mu={mu}, var={var}".format(name = name[k], mu = format(mu,".2f"), var = format(sigma ** 2,".2f")))
    plt.plot(bins, best_fit_line)
    k += 1
#plt.title("Conditioning: S > 0.5")
plt.show()

corr = df.corr()
print(corr)
print(df.describe())

"""
def MI(x,y,Nbins=21):
    bins = np.linspace(np.min(x),np.max(x),Nbins)
    eps=np.spacing(1)
    x_marginal = np.histogram(x,bins=bins)[0]
    x_marginal = x_marginal/x_marginal.sum()
    y_marginal = np.array(np.histogram(y,bins=bins)[0])
    y_marginal = y_marginal/y_marginal.sum()
    xy_joint = np.array(np.histogram2d(x,y,bins=(bins,bins))[0])
    xy_joint = xy_joint/xy_joint.sum()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(xy_joint.T,origin='lower')
    plt.title('joint')
    plt.subplot(1,2,2)
    plt.imshow((x_marginal[:,None]*y_marginal[None,:]).T,origin='lower')
    plt.title('product of marginals')
    MI=np.sum(xy_joint*np.log(xy_joint/(x_marginal[:,None]*y_marginal[None,:]+eps)+eps))
    plt.suptitle('Mutual information: %f'%MI)
    return(MI)
#MI(df["R"].values, df["S"].values, 21)
print(np.corrcoef(df["R"].values, df["S"].values))
"""