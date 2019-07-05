import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib as mpl
import matplotlib.pyplot as plt

agree_weight = []
df = pd.read_csv('students.csv')
df.columns = ['TS', 'TM1', 'TM2', 'PC', 'Resources', 'CH1', 'CH2', 'A1', 'A2', 'Power', 'Q1', 'Q2', 'PD1', 'PD2', 'RT',
              'RT2', 'RS1', 'RS2', 'AF1', 'AF2', 'c1', 'c2', 'RD1', 'RD2', 'Skills', 'r', 'con1', 'con2', 'support',
              'ch', 'change']

stud_pre_df = [['TM1', 'con1', 'PC', 'CH1', 'A1', 'AF1', 'Power', 'Q1',
                'PD1', 'c1', 'RD1', 'support']].copy()
stud_post_df = [['TM2', 'con2', 'PC', 'CH2', 'A2', 'AF2', 'Power', 'Q2',
                'PD2', 'c2', 'RD2', 'support']].copy()

def get_individual_agree_weight(i):
    if i.casefold() == 'strongly agree' or i=='yes':
        temp =1
    elif i.casefold() == 'agree':
        temp = 0.75
    elif i.casefold() == 'neutral' or i=='Cant say':
        temp = 0.5
    elif i.casefold() == 'strongly disagree' or i=='no':
        temp = 0
    elif i.casefold() == 'disagree':
        temp = 0.25
    else:
        temp = 0
    return temp

def get_agree_weight(var):
    agree_weight_list = []
    for i in var:
        if i.casefold() == 'strongly agree' or i=='yes':
            agree_weight_list.append(1)
        elif i.casefold() == 'agree':
            agree_weight_list.append(0.75)
        elif i.casefold() == 'neutral' or i=='Cant say':
            agree_weight_list.append(0.5)
        elif i.casefold() == 'strongly disagree' or i=='no':
            agree_weight_list.append(0)
        elif i.casefold() == 'disagree':
            agree_weight_list.append(0.25)
        else:
            # agree_weight_list.append(TextBlob(i).sentiment.polarity)
            agree_weight_list.append(0)

    #print(agree_weight_list)
    return agree_weight_list


def get_agree_avg(dict_name):
    agree_weight_list = get_agree_weight(dict_name)
    list_name = []

    i = 0
    for key in dict_name:
        if agree_weight_list[i] == 0:
            i = i
        else:
            list_name.append(agree_weight_list[i] * dict_name[key])
        i = i + 1

    #print(list_name)
    return sum(list_name) / sum(dict_name.values())



#  ---------------------------------------Atrributes Calculations start---------------------------- #




# Teaching Methologies AND Content Delivery
teaching_mean_pre = (df['TM1'].mean() + df['con1'].mean()) / 20
teaching_mean_post = (df['TM2'].mean() + df['con2'].mean()) / 20
# print("teaching_mean_pre: ", teaching_mean_pre)
# print("teaching_mean_post: ", teaching_mean_post)

# Participation
# Handle comments
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
#print(participation_weight)






# College Hours
# Have to eliminate the neutral values - Incomplete
clgHrs_pre = df['CH1'].apply(lambda x: TextBlob(x).sentiment.polarity)
clgHrs_pre = clgHrs_pre.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print (clgHrs_pre)
clgHrs_post = df['CH2'].apply(lambda x: TextBlob(x).sentiment.polarity)
clgHrs_post = clgHrs_post.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print (clgHrs_post)


#   Attendance
attend_pre = df['A1'].value_counts().to_dict()
attendance_pre = (get_agree_avg(attend_pre) + (df['AF1'].mean() / 10)) / 2
#print (attendance_pre)

# print("Post Autonomy")
attend_post = df['A2'].value_counts().to_dict()
attendance_post = (get_agree_avg(attend_post) + (df['AF2'].mean() / 10)) / 2
#print (attendance_post)



# Biased Behaviour
dictionary = df['Power'].value_counts().to_dict()
#biased_post = get_agree_avg(dictionary)
weight_list = []
for i in dictionary:
    if i.casefold() == 'strongly agree':
        weight_list.append(0)
    elif i.casefold() == 'agree':
        weight_list.append(0.25)
    elif i.casefold() == 'neutral' or i == 'Cant say':
        weight_list.append(0.5)
    elif i.casefold() == 'strongly disagree' or i == 'no':
        weight_list.append(1)
    elif i.casefold() == 'disagree':
        weight_list.append(0.75)


biased_post =0
i=0
for key in dictionary:
    biased_post = (weight_list[i]*dictionary[key])+biased_post
    i+=1
biased_post/=sum(dictionary.values())
#print(dictionary)
#print(biased_post)
biased_pre = 1


# Quality of Cirriculum
quality_pre = df['Q1'].mean()/10
#print(quality_pre)
quality_post = df['Q2'].mean()/10
#print(quality_post)




# Personal Development
personalDevelopment_pre = df['PD1'].apply(lambda x: TextBlob(x).sentiment.polarity)
personalDevelopment_pre = personalDevelopment_pre.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print(personalDevelopment_pre)
personalDevelopment_post = df['PD2'].apply(lambda x: TextBlob(x).sentiment.polarity)
personalDevelopment_post = personalDevelopment_post.apply(lambda x: (0.5-(-x*0.5)) if x<0 else (0.5+(x*0.5)) ).mean()
#print(personalDevelopment_post)


# Mental Support
variable = df['c1'].value_counts().to_dict()
support_pre = ((df['RD1'].mean()/10) + (df['support'].mean()/10) + get_agree_avg(variable))/3
#print(support_pre)

variable2 = df['c2'].value_counts().to_dict()
support_post = ((df['RD2'].mean()/10) + (df['support'].mean()/10) + get_agree_avg(variable2))/3
#print(support_post)


#   Final Satisfaction level Calculation
weights = [0.3, 0.23, 0.17, 0.09, 0.08, 0.08, 0.03, 0.02]
attributes_pre = [personalDevelopment_pre, teaching_mean_pre, quality_pre, participation_weight, clgHrs_pre,
                  support_pre, attendance_pre, biased_pre]
attributes_post = [personalDevelopment_post, teaching_mean_post, quality_post, participation_weight, clgHrs_post,
                  support_post, attendance_post, biased_post]
satisfaction_pre = list(map ( (lambda x,y: x*y), weights, attributes_pre) )
satisfaction_pre = sum(satisfaction_pre)
#print(satisfaction_pre)

satisfaction_post = list(map ( (lambda x,y: x*y), weights, attributes_post) )
satisfaction_post = sum(satisfaction_post)
#print(satisfaction_post)



#----------------------------Visualization Part -----------------------------------#
def satisfaction_graph():
    mpl.get_backend()
    pre = attributes_pre
    post = attributes_post

    r = np.array([satisfaction_pre, satisfaction_post])
    ind = np.arange(8)
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, pre, width, label="Pre-Autonomy")
    rects2 = ax.bar(ind + width, post, width, label="Post-Autonomy")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Personal\nDevelopment', 'Teaching\nMethodologies', 'Quality of\nCirriculum',
                        'Participation', 'College\nHours', 'Mental\nSupport',
                        'Attendance', 'Unbiaseness'))

    plt.subplots_adjust(bottom=0.25)
    x = plt.gca().xaxis
    for i in x.get_ticklabels():
        i.set_rotation(45)
    plt.legend()
    plt.show()




ind_sat_pre=[]
ind_sat_post=[]
def satisfaction_stud_graph():
    for i in range(df.shape[0]):
        clgHr = TextBlob(df['CH1'].iloc[i]).sentiment.polarity
        #clgHr = (lambda x: (0.5 - (-x * 0.5)) if x < 0 else (0.5 + (x * 0.5)), clgHr)
        if clgHr<0:
            clgHr = 0.5 - (-clgHr * 0.5)
        else:
            clgHr = 0.5 + (clgHr * 0.5)

        attend = get_individual_agree_weight(df['A1'].iloc[i])

        variable2 = TextBlob(df['PD1'].iloc[i]).sentiment.polarity
        #variable2 = (lambda x: (0.5 - (-x * 0.5)) if x < 0 else (0.5 + (x * 0.5)), variable2)
        if variable2<0:
            variable2 = 0.5 - (-variable2 * 0.5)
        else:
            variable2 = 0.5 + (variable2 * 0.5)

        variable3 = TextBlob(df['c1'].iloc[i]).sentiment.polarity
        #variable3 = (lambda x: (0.5 - (-x * 0.5)) if x < 0 else (0.5 + (x * 0.5)), variable3)
        if variable3<0:
            variable3 = 0.5 - (-variable3 * 0.5)
        else:
            variable3 = 0.5 + (variable3 * 0.5)

        #variable4 = ((df['TM1'].iloc[i]/10) + clgHr + attend + (df['AF1'].iloc[i]/10) + 1 + (df['Q1'].iloc[i]/10) + variable2 + variable3 + (df['RD1'].iloc[i]/10) + (df['support'].iloc[i]/10))
        variable4 = ( (variable2*0.3) + ( ((df['con1'].iloc[i]/10)+(df['TM1'].iloc[i]/10)/2)*0.23 )  + ( (df['Q1'].iloc[i]/10)*0.17 ) + ( clgHr*0.08 ) + ( (((variable3+(df['RD1'].iloc[i]/10) + (df['support'].iloc[i]/10)))/3)*0.08 ) + ( (attend+(df['AF1'].iloc[i]/10)/2)*0.03 ) + (1*0.02) )
        #print(stud_pre)
        ind_sat_pre.append(variable4)

        #Post Auutonomy
        clgHr = TextBlob(df['CH2'].iloc[i]).sentiment.polarity
        # clgHr = (lambda x: (0.5 - (-x * 0.5)) if x < 0 else (0.5 + (x * 0.5)), clgHr)
        if clgHr < 0:
            clgHr = 0.5 - (-clgHr * 0.5)
        else:
            clgHr = 0.5 + (clgHr * 0.5)

        attend = get_individual_agree_weight(df['A2'].iloc[i])

        variable2 = TextBlob(df['PD2'].iloc[i]).sentiment.polarity
        # variable2 = (lambda x: (0.5 - (-x * 0.5)) if x < 0 else (0.5 + (x * 0.5)), variable2)
        if variable2 < 0:
            variable2 = 0.5 - (-variable2 * 0.5)
        else:
            variable2 = 0.5 + (variable2 * 0.5)

        variable3 = TextBlob(df['c2'].iloc[i]).sentiment.polarity
        # variable3 = (lambda x: (0.5 - (-x * 0.5)) if x < 0 else (0.5 + (x * 0.5)), variable3)
        if variable3 < 0:
            variable3 = 0.5 - (-variable3 * 0.5)
        else:
            variable3 = 0.5 + (variable3 * 0.5)

        if df['Power'].iloc[i].casefold() == 'strongly agree':
            temp = 0
        elif df['Power'].iloc[i].casefold() == 'agree':
            temp = 0.25
        elif df['Power'].iloc[i].casefold() == 'neutral' or df['Power'].iloc[i].casefold() == 'Cant say':
            temp = 0.5
        elif df['Power'].iloc[i].casefold() == 'strongly disagree' or df['Power'].iloc[i].casefold() == 'no':
            temp = 1
        elif df['Power'].iloc[i].casefold() == 'disagree':
            temp = 0.75

        variable4 = ((variable2 * 0.3) + ( ((df['con2'].iloc[i]/10) + (df['TM2'].iloc[i]/10)/2)*0.23 ) + ( (df['Q2'].iloc[i]/10)*0.17 ) + (clgHr * 0.08) + ( (((variable3 + (df['RD2'].iloc[i] / 10) + (df['support'].iloc[i]/10)))/3)*0.08 ) + ( (attend + (df['AF2'].iloc[i]/10)/2)*0.03 ) + (temp*0.02) )
        ind_sat_post.append(variable4)


def ind_sat_graph():
    satisfaction_stud_graph()
    plt.scatter(ind_sat_pre, ind_sat_post)
    plt.xlabel('Pre-Autonomy', fontsize=20)
    plt.ylabel('Post-Autonomy', fontsize=20)
    plt.show()

#ind_sat_graph()