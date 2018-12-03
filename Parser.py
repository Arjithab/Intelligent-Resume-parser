#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv as csv
import pandas as pd
import re

#data=pd.read_csv('Globalops.csv')
#data=pd.read_csv('Globalops1770.csv')
data=pd.read_csv("C:/Python27/Data Dic/Globalops50.csv")      # INPUt File path
id_rtf=data[['CandidateId','resumeRTF']]
mydict = id_rtf.set_index('CandidateId')['resumeRTF'].to_dict()


    
def find_email(text):
    match = re.search(r'([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4})',text,re.I)
    
    if match:
        mail=match.group(0)
        return mail
    else:
        mail="NULL"
        return mail
def find_experience(text):
    match=re.search(r'(?:\d?|\d\d?)+\s+(?:years?|yrs?)\s+(?:and\s*)?(?:\d?|\d\d?)+\s+months?|(?:\d?|\d\d?|\d\d\d?)+\+(?:months?|years?|yrs?)|(?:\d?|\d\d?)\.?(?:\d?|\d\d?)+\s+(?:yrs?|years?|months)|(?:\d?|\d\d?)\.?(?:\d?|\d\d?)(?:\+?)+\s+(?:yrs?|years?|months)|(?:\d?|\d\d?|\d\d\d?)(?:\+?)+\s+(?:months?|years?|yrs?)',d[k],re.I)
    
    if match:
        experience=match.group(0)
        return experience
    else:
        experience="NULL"
        return experience
    
def find_institute(text):
    institute_df=pd.read_csv('C:/Python27/Data Dic/institution.csv')    # Institute Dictionary path
    institute_dic = institute_df.set_index('Institute')['SlNo'].to_dict()
    institute_list=institute_dic.keys()
    institute_split = [key.split(',') for key in institute_list]
    
    POSITION=['NULL']
    
    for i in range(0,len(institute_split)):
        x=str(institute_split[i])[2:-2].lower()
        if x in str(text).lower():
            part=str(institute_split[i])[2:-2]
            POSITION.append(part)
            y='NULL'
            if y in POSITION:
                POSITION.remove('NULL')
                
    for i in range(0,(5-len(POSITION))):
        a='NULL'
        POSITION.append(a)       
            
    return POSITION 

def find_education(text):
    education_df=pd.read_csv('C:/Python27/Data Dic/education.csv')   # Education Dictionary path
    education_dic = education_df.set_index('Course')['SlNo'].to_dict()
    education_list=education_dic.keys()
    education_split = [key.split(',') for key in education_list]
    
    POSITION=['NULL']
    
    for i in range(0,len(education_split)):
        x=str(education_split[i])[2:-2].lower()
        if x in str(text).lower():
            part=str(education_split[i])[2:-2]
            POSITION.append(part)
            y='NULL'
            if y in POSITION:
                POSITION.remove('NULL')
    for i in range(0,(5-len(POSITION))):
        a='NULL'
        POSITION.append(a)      
            
    return POSITION
    
def find_position(text):
    position_df=pd.read_csv('C:/Python27/Data Dic/positions.csv')      # Position Dictionary path
    position_dic = position_df.set_index('JobTitle')['SlNo'].to_dict()
    position_list=position_dic.keys()
    position_split = [key.split(',') for key in position_list]
    
    POSITION=['NULL']
    
    for i in range(0,len(position_split)):
        x=str(position_split[i])[2:-2].lower()
        if x in str(text).lower():
            part=str(position_split[i])[2:-2]
            POSITION.append(part)
            y='NULL'
            if y in POSITION:
                POSITION.remove('NULL')
                
    for i in range(0,(5-len(POSITION))):
        a='NULL'
        POSITION.append(a)      
            
    return POSITION 

def find_skill(text):
    skill_df=pd.read_csv('C:/Python27/Data Dic/skills.csv')   # Skill Dictionary path
    skill_dic = skill_df.set_index('JobTitle')['SlNo'].to_dict()
    skill_list=skill_dic.keys()
    skill_split = [key.split(',') for key in skill_list]
    
    SKILL=['NULL']
    
    for i in range(0,len(skill_split)):
        x=str(skill_split[i])[2:-2].lower()
        if x in str(text).lower():
            part=str(skill_split[i])[2:-2]
            SKILL.append(part)
            y='NULL'
            if y in SKILL:
                SKILL.remove('NULL')
               
            
    for i in range(0,(15-len(SKILL))):
        a='NULL'
        SKILL.append(a)            
             
    return SKILL    
    
def name_the_key(dict, key):
    return key, dict[key]


def print_id_email(mydict,key_pass):      # Row data format
    key_name, value = name_the_key(mydict, key_pass)
    EMAILID=find_email(str(value))
    EXPERIENCE=find_experience(str(value))
    EDUCATION=find_education(value)
    INSTITUTE=find_institute(value)
    POSITION= find_position(value)
    SKILL= find_skill(value)
    
    ALL= str(key_name)+':'+EMAILID+':'+str(POSITION[0])+':'+str(POSITION[1])+':'+str(POSITION[2])+':'+str(POSITION[3])+':'+str(POSITION[4])+':'+str(SKILL[0])+':'+str(SKILL[1])+':'+str(SKILL[2])+':'+str(SKILL[3])+':'+str(SKILL[4])+':'+str(SKILL[5])+':'+str(SKILL[6])+':'+str(SKILL[7])+':'+str(SKILL[8])+':'+str(SKILL[9])+':'+str(SKILL[10])+':'+str(SKILL[11])+':'+str(SKILL[12])+':'+str(SKILL[13])+':'+str(SKILL[14])+':'+str(EDUCATION[0])+':'+str(EDUCATION[1])+':'+str(EDUCATION[2])+':'+str(EDUCATION[3])+':'+str(EDUCATION[4])+':'+str(INSTITUTE[0])+':'+str(INSTITUTE[1])+':'+str(INSTITUTE[2])+':'+str(INSTITUTE[3])+':'+str(INSTITUTE[4])+':'+EXPERIENCE 
    #print(ALL)
    return ALL


w = csv.writer(open("C:\Python27\output.csv", "w")) # Output file path
#w = csv.writer(open("output.csv", "w"))



Heading=["Candidate ID:EmailID:Position_1:Position_2:Position_3:Position_4:Position_5:Skill_1:Skill_2:Skill_3:Skill_4:Skill_5:Skill_6:Skill_7:Skill_8:Skill_9:Skill_10:Skill_11:Skill_12:Skill_13:Skill_14:Skill_15:Education_1:Education_2:Education_3:Education_4:Education_5:Institute_1:Institute_2:Institute_3:Institute_4:Institute_5:Experience"]
w.writerow(Heading)
print(Heading)

for key in mydict.keys():
    insert_row=str(print_id_email(mydict,key))
    w.writerow([insert_row])
    print(insert_row)
    

##shortlisted resumes based on skill and experience.
import pandas as pd
import csv as csv
import re

lskill=['java','ui']
lexp=3
d5=dict()
d4=dict()
d2=dict()
d3=dict()
d6=dict()
d8=dict()

def resume_finder_skills(d1):
    count=0
    w = csv.writer(open("C:\Python27\ds.csv", "w"))  ##shortlisted resumes
    w.writerow(['CandidateId','skill'])
    w1 = csv.writer(open("C:\Python27\interview_skills.csv","w"))  ##integer values with count weightage
    for k in d1.keys():
        count=0
        for l in lskill:
            if l in d1[k]:
                count=count+1
                
        if count<=len(lskill) and count!=0:
            w.writerow([k,d1[k]])
            
            if 'ui'in d1[k] and 'java' in d1[k]:
                w1.writerow([k,d1[k].count('java'),d1[k].count('ui')])
                d4.update({k:[d1[k].count('java'),d1[k].count('ui')]})
            elif 'ui' in d1[k] and 'java' not in d1[k]:
                w1.writerow([k,0,d1[k].count('ui')])
                d4.update({k:[0,d1[k].count('ui')]})
            else:
                w1.writerow([k,d1[k].count('java'),0])
                d4.update({k:[d1[k].count('java'),0]})
   
               
def resume_finder_exp():
    
    w2= csv.writer(open("C:\Python27\shortlisted.csv","w"))
    w2.writerow(['CandidateId','EmailId','Experience','Java','UI','Skillset','Positions held'])
    for k in d3.keys():
        if k in d4.keys():
            z=str(d3[k]).split()
            if z[0].replace('.','',1).isdigit() == True:
                if float(z[0])>=float(3):
                    d5.update({k:z[0]})
                    w2.writerow([k,d6[k],z[0],d4[k][0],d4[k][1],d2[k],d8[k]])
                    
                    
def retrieve_positions(k,d):
    
    f=open("C:/Python27/positions.txt")
    data=f.read()
    data=data.lower()
    words=data.split("\n")
    positions=list(words)
    v=d[k]
    v=str(v)
    v=v.lower()
    d7=dict()
    d7[k]=list()
    for l in positions:
        if l in v:
            if l!=' ' and l!='':
                d7[k].append(l)
    for k in d7.keys():
        s=set(d7[k])
        d7[k]=list(s)
    ##print(d1)
    d8.update(d7)
    
  
def retrieve_skills(k,d):
    
    f=open("C:/Python27/skills.txt")
    data=f.read()
    data=data.lower()
    words=data.split("\n")
    skills=list(words)
    v=d[k]
    v=str(v)
    v=v.lower()
    d1=dict()
    d1[k]=list()
    for l in skills:
        if l in v:
            if l!=' ' and l!='':
                d1[k].append(l)
    for k in d1.keys():
        s=set(d1[k])
        d1[k]=list(s)
    ##print(d1)
    d2.update(d1)
   

def retrieve_exp(d):
    for k in d.keys():
        match=re.search(r'(?:\d?|\d\d?)+\s+(?:years?|yrs?)\s+(?:and\s*)?(?:\d?|\d\d?)+\s+months?|(?:\d?|\d\d?|\d\d\d?)+\s+(?:months?|years?|yrs?)|(?:\d?|\d\d?)\.?(?:\d?|\d\d?)+\s+(?:yrs?|years?|months)|(?:\d?|\d\d?)\.?(?:\d?|\d\d?)(?:\+?)+\s+(?:yrs?|years?|months)|(?:\d?|\d\d?|\d\d\d?)(?:\+?)+\s+(?:months?|years?|yrs?)',str(d[k]),re.I)
        if match:
            experience=match.group(0)
        else:
            experience="NULL"
        d3.update({k:str(experience)})
        

def retrieve_email(d):
    for k in d.keys():
        match = re.search(r'([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4})',str(d[k]),re.I)
        if match:
            mail=match.group(0)
        else:
            mail="NULL"        
        d6.update({k:str(mail)})  
    print d6
    
def main():  
    f=pd.read_csv("C:/Python27/aa.csv")
    df=f[['CandidateId','resumeRTF']]
    d=df.set_index('CandidateId')['resumeRTF'].to_dict() ##dict with id and resume
    for key in list(d.keys()):  ## remove null values
        if d[key] == 'Nan' or d[key] == 'nan':
            del d[key]
        
    print("Extracting Skills :")
    for k in d.keys():
        retrieve_skills(k,d)
    resume_finder_skills(d2)   
      
    print("Extracting E-mail ID :")
    retrieve_email(d)
    
    print("Extracting positions :")
    for k in d.keys():
        retrieve_positions(k,d)
        
    print("Extracting Experience :")
    retrieve_exp(d)
    resume_finder_exp()
    
    print("Done.")
        
if __name__== "__main__":
    main()
    

##classification done on shortlisted resumes

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from mlxtend.plotting import plot_decision_regions
from mlxtend.preprocessing import shuffle_arrays_unison
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from mlxtend.plotting import plot_confusion_matrix

df=pd.read_csv("C:\Python27\shortlisted.csv")
d1=df[['Measure']]
d1=d1.iloc[::2]
X=np.array(d1)
y=[5]*25
y=y+([1]*91)
y=y+([2]*75)
y=y+([20]*191)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
 
# accuracy on X_test
accuracy = gnb.score(X_test, y_test)
print accuracy
 
# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)
print cm
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()

#training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print accuracy
 
# creating a confusion matrix
knn_predictions = knn.predict(X_test) 
cm = confusion_matrix(y_test, knn_predictions)
print cm
 
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
print accuracy
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print cm

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
print cm
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set

from sklearn.model_selection import cross_val_score

#score = cross_val_score(regressor, X_test, y_test)
score = regressor.score(X_test, y_test)

print score  # python 2.x 
y=np.array(y)
plot_decision_regions(X, y, clf=svm_model_linear, res=0.02,
                      legend=2, X_highlight=X_test)

# Adding axes annotations
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Resume classification')
plt.show()

