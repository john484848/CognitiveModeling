import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
charachters=["Ronald McDonald", "Mickey Mouse"]
charachtersp=np.array([0.3,0.7])
weapons=["Gloves", "Fork and Knife", "Dirty Bib"]
weaponmp=np.array([0.67,0.13,0.20,0.21,0.29,0.5])
predicted=[]
def choseweapon(charindex):
    return np.random.choice(weapons,p=weaponmp[charindex*3:charindex*3+3])
def makestr(charindex,weapon):
    return charachters[charindex]+" - "+weapon
def convertp(count,numsim):
    return count/numsim
vfunc=np.vectorize(choseweapon)
vfunc2=np.vectorize(makestr)
vfunc3=np.vectorize(convertp)
for i in range(len(charachters)):
    for l in range(3*i,3*i+3):
        predicted.append((charachters[i]+" - "+ weapons[l%3],weaponmp[l]*charachtersp[i]))
for i in predicted:
    print(i[0],i[1])
globalres=[]
for i in range(1,10000):
    charchoice=np.random.choice(np.arange(charachtersp.size),size=i,p=charachtersp)
    weaponl=vfunc(charchoice)
    resultstr=vfunc2(charchoice,weaponl)
    outcome,counts=np.unique(resultstr,return_counts=True)
    counts=vfunc3(counts,numsim=i)
    diff=0
    for e in range(len(predicted)):
        index=-1
        for r in range(len(outcome)):
            if outcome[r]==predicted[e][0]:
                index=r
                break
        if index==-1:
            diff+=predicted[e][1]
            continue
        diff+=abs(predicted[e][1]-counts[r])
    globalres.append((i,diff))
df=pd.DataFrame(globalres,columns=["Num sims","Probability Diff"])
sns.set_theme()
sns.relplot(data=df, x="Num sims", y="Probability Diff", kind="line")
plt.show()
