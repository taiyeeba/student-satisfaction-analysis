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


# Teaching Methologies AND Content Delivery
teaching_pre = df['TM1'].mean()/10
teaching_post = df['TM2'].mean()/10
print("teaching_pre: ", teaching_pre)
print("teaching_post: ", teaching_post)

#  Content Delivery
content_pre = df['con1'].mean()/10
content_post = df['con2'].mean()/10
#print(content_pre)
#print(content_post)

# Quality of Cirriculum
quality_pre = df['Q1'].mean()/10
#print(quality_pre)
quality_post = df['Q2'].mean()/10
#print(quality_post)

#  Personal Development
personalDevelopment_pre = df['PD1'].apply(lambda x: TextBlob(x).sentiment.polarity)
personalDevelopment_pre = personalDevelopment_pre.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print(personalDevelopment_pre)
personalDevelopment_post = df['PD2'].apply(lambda x: TextBlob(x).sentiment.polarity)
personalDevelopment_post = personalDevelopment_post.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print(personalDevelopment_post)



#  Final QualtiyOfEducation Calculation
weights = [0.4, 0.3, 0.2, 0.1]
attributes_pre = [quality_pre, teaching_pre, content_pre, personalDevelopment_pre]
attributes_post = [quality_post, teaching_post, content_post, personalDevelopment_post]
eduQuality_pre = list(map ( (lambda x,y: x*y), weights, attributes_pre) )
eduQuality_pre = sum(eduQuality_pre)
print(eduQuality_pre)

eduQuality_post = list(map ( (lambda x,y: x*y), weights, attributes_post) )
eduQuality_post = sum(eduQuality_post)
print(eduQuality_post)

#----------------------------Visualization Part -----------------------------------#
def quality_graph():
    mpl.get_backend()
    pre = attributes_pre
    post = attributes_post

    r = np.array([eduQuality_pre, eduQuality_post])
    ind = np.arange(4)
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, pre, width, label="Pre-Autonomy")
    rects2 = ax.bar(ind + width, post, width, label="Post-Autonomy")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Qualtiy Of\nCirriculum', 'Teaching\nMethodologies',
                        'Content\nDelivery', 'Personal\nDevelopment'))

    plt.subplots_adjust(bottom=0.25)
    x = plt.gca().xaxis
    for i in x.get_ticklabels():
        i.set_rotation(45)
    plt.legend()
    plt.show()