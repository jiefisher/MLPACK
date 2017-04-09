'''
Created on 2017/4/9/

@author: jiefisher
'''
import numpy as np
import math
import pandas as pd
import lr as lgr
if __name__ == '__main__':
    
    lr=lgr.LogisticRegression(learning_rate=.0001,l2=.0005,iters=50000)
    df=pd.read_csv("train.csv")
    x_label=df["Survived"]
    df=df.drop(["Survived","PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)
    df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
    df=df.fillna(0)
    lr.train(df.values, x_label.values)
    df=pd.read_csv("test.csv")
    bf=pd.read_csv("gender_submission.csv")
    x_label=bf["Survived"]
    df=df.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)
    df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
    df=df.fillna(0)
    print(lr.predict(df.values))