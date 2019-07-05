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
