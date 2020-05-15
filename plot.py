import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy import stats
#%matplotlib inline
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs

X,y= make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='summer');

xfit=np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c=y ,s=50, cmap='summer')
plt.plot([0.6],[2.1],'x',color='green', markeredgewidth=2, markersize=5)

for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit, m*xfit+b, '-k')

plt.xlim(-1,3.5)
xfit=np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c=y ,s=50, cmap='summer')
for m,b,d in[(1,0.65,0.2),(0.5,1.6,0.2),(-0.2,2.9,0.2)]:
    yfit=m*xfit+b
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit,yfit-d,yfit+d, edgecolor='none',color='#AAAAED',alpha=0.4)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>Visualizing decision boundaries
#model=SVC(kernel='linear', c=1E10)
#model.fit(X,y)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




plt.scatter(X[:,0],X[:,1], c=y,s=50,cmap='summer')
plot_svc_decision_function(model);
plt.xlim(-1,3.5)

plt.show()
