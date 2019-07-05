import config
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('students.csv')
df.columns = ['TS', 'TM1', 'TM2', 'PC', 'Resources', 'CH1', 'CH2', 'A1', 'A2', 'Power', 'Q1', 'Q2', 'PD1', 'PD2', 'RT',
              'RT2', 'RS1', 'RS2', 'AF1', 'AF2', 'c1', 'c2', 'RD1', 'RD2', 'Skills', 'r', 'con1', 'con2', 'support',
              'ch', 'change']


#----------------------------Attributes Calculation Start---------------------#


# Participation
participation = df.PC.str.strip()
participation = participation.str.replace(", ", ",")
participation = pd.DataFrame(participation.str.split(",").tolist()).stack()
p = participation.value_counts().to_dict()
#print(p)
neg_bag = ['Attendance', 'No time', 'There are no returns/rewards', 'Not Interested']
pos_bag = ['None of the above, as I will participate any way']
neg_weight = 0
pos_weight = 0
neg = 1/len(neg_bag)
pos = 1/len(pos_bag)
ncount=0
pcount=0
count = 0
for key in p:
    if key in neg_bag:
        count += p[key]
        ncount+=p[key]
    elif key in pos_bag:
        count += p[key]
        pcount+=p[key]
#    else:
#        print (key, " : ", TextBlob(key).sentiment.polarity)
#print(ncount)
#print(count)
if ncount==0:
    neg_weight=0
else:
    neg_weight = 0.5 - ((ncount/count) * 0.5)

if pcount==0:
    pos_weight=0
else:
    pos_weight = 0.5 + ((pcount/count)*0.5)

participation_weight = neg_weight + pos_weight
print(participation_weight)



#  Free Time i.e College Hours + Attendance
clgHrs_pre = df['CH1'].apply(lambda x: TextBlob(x).sentiment.polarity)
clgHrs_pre = clgHrs_pre.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print (clgHrs_pre)
clgHrs_post = df['CH2'].apply(lambda x: TextBlob(x).sentiment.polarity)
clgHrs_post = clgHrs_post.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print (clgHrs_post)

attend_pre = df['A1'].value_counts().to_dict()
attendance_pre = (config.get_agree_avg(attend_pre) + (df['AF1'].mean() / 10)) / 2
#print (attendance_pre)
attend_post = df['A2'].value_counts().to_dict()
attendance_post = (config.get_agree_avg(attend_post) + (df['AF2'].mean() / 10)) / 2
#print (attendance_post)

freeTime_pre = (clgHrs_pre+attendance_pre)/2
freeTime_post = (clgHrs_post+attendance_post)/2
print(freeTime_pre)
print(freeTime_post)




# Support to try something new, creative ideas & Research n Development
variable = df['c1'].value_counts().to_dict()
support_pre = ((df['RD1'].mean()/10) + config.get_agree_avg(variable))/2
print(support_pre)

variable2 = df['c2'].value_counts().to_dict()
support_post = ((df['RD2'].mean()/10)  + config.get_agree_avg(variable2))/2
print(support_post)


#  Final Student Development Calculation
weights = [0.5, 0.3, 0.2]
attributes_pre = [support_pre, participation_weight, freeTime_pre]
attributes_post = [support_post, participation_weight, freeTime_post]
studentDevelopment_pre = list(map ( (lambda x,y: x*y), weights, attributes_pre) )
studentDevelopment_pre = sum(studentDevelopment_pre)
print(studentDevelopment_pre)

studentDevelopment_post = list(map ( (lambda x,y: x*y), weights, attributes_post) )
studentDevelopment_post = sum(studentDevelopment_post)
print(studentDevelopment_post)

#----------------------------Visualization Part -----------------------------------#
def studDev_graph():
    mpl.get_backend()
    pre = attributes_pre
    post = attributes_post

    r = np.array([studentDevelopment_pre, studentDevelopment_post])
    ind = np.arange(3)
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, pre, width, label="Pre-Autonomy")
    rects2 = ax.bar(ind + width, post, width, label="Post-Autonomy")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Support', 'Participation', 'Free Time'))

    plt.subplots_adjust(bottom=0.25)
    x = plt.gca().xaxis
    for i in x.get_ticklabels():
        i.set_rotation(45)
    plt.legend()
    plt.show()