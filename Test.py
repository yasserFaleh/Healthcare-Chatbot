import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree ,plot_tree,export_graphviz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import re
import nltk
from nltk.stem import WordNetLemmatizer
import random


lemmatizer = WordNetLemmatizer()
training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
symptoms=dict()

symptoms_dict = {}

predected_diseases = []
symptoms_present = []
diseases = {}
precution_list = []

rejected={"pain"}

greeting_inputs = ("hey","Hi","Hey","How are you","Is anyone there?","Hello","Good day","Hello","How are you","good morning","Is anyone there?", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey","Hey :-)", "hey hows you?", "hello, how you doing", "hello", "Welcome","Hello, thanks for visiting","Hi there, what can I do for you?","Hi there, how can I help?"]

goodbye_inputs=( "Bye", "bye","goodbye","good","Good","See you later", "Goodbye" )
goodbye_responses = ["See you later, thanks for visiting","Have a nice day","Bye! Come back again soon." ]

thanks_inputs =("Thanks", "thanks","Thank","thank","Thank you", "That's helpful", "Thank's a lot!")
thanks_responses =["Happy to help!", "Any time!", "My pleasure" ]


def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return True,random.choice(greeting_responses)
        else:
            return False,"0"

def generate_goodbye_response(goodbye):
    for token in goodbye.split():
        if token.lower() in goodbye_inputs:
            return True,random.choice(goodbye_responses)
        else:
            return False,"0"

def generate_thanks_response(thanks):
    for token in thanks.split():
        if token.lower() in thanks_inputs:
            return True,random.choice(thanks_responses)
        else:
            return False,"0"



def init():
    #Read from excell files
    desc =open('Description.csv')
    severity=open('severity.csv')
    precaution=open('precaution.csv')
    csv_reader_desc = csv.reader(desc, delimiter=',')
    csv_reader_severity = csv.reader(severity, delimiter=',')
    csv_reader_precaution = csv.reader(precaution, delimiter=',')
    # Extract data into dicts
    for row in csv_reader_desc:
        _description = {row[0]: row[1]}
        description_list.update(_description)
    for row in csv_reader_precaution:
        prec = {row[0]: [row[1], row[2], row[3], row[4]]}
        precautionDictionary.update(prec)
    for index, symptom in enumerate(x):
        symptoms_dict[symptom] = index
    try:
        for row in csv_reader_severity:
            _diction = {row[0]: int(row[1])}
            severityDictionary.update(_diction)
    except:
        pass


def check_pattern(dis_list, input): # Comparing a word in the words list
    pred_list=[]
    regexp = re.compile(input)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,item

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {}
    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1
    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease

def recurse(node, depth,Input,feature_names,tree):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        if name == Input:
            val = 1
        else:
            val = 0
        if  val <= threshold:
            recurse(tree_.children_left[node], depth + 1,Input,feature_names,tree)
        else:
            symptoms_present.append(name)
            recurse(tree_.children_right[node], depth + 1,Input,feature_names,tree)
    else:
        present_disease = print_disease(tree_.value[node])
        red_cols = reduced_data.columns
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
        symptoms_exp=[]
        inp = ""
        for syms in list(symptoms_given):
            if syms in symptoms.keys() :
                if (symptoms[syms] == "yes"):
                    symptoms_exp.append(syms)
            else:
                print(syms,"? : ",end='')
                while True:
                    inp = input("")
                    if (inp == "yes" or inp == "no"):
                        symptoms[syms]=inp
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ", end="")
                if (inp == "yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
        if(present_disease[0]==second_prediction[0]):
            predected_diseases.append(present_disease[0])
        else:
            predected_diseases.append(present_disease[0])
            predected_diseases.append(second_prediction[0])

def tree_to_code(tree, feature_names):
    diseases=[]
    global predected_diseases

    chk_dis = ",".join(feature_names).split(",")


    print("Enter the symptom you are experiencing  \n\t\t\t\t\t\t", end="->")
    disease_input = input("")
    tokens = nltk.word_tokenize(disease_input)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in rejected]
    for token in tokens:
        isGreeted,greeting_response = generate_greeting_response(token)
        if(isGreeted):
            print(greeting_response)

    tokens = sorted(set(tokens))

    tokens = [w for w in tokens if len(w) > 3]
    for w in tokens:
        conf, cnf_dis = check_pattern(chk_dis, w)  # conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf == 1:
            for num, it in enumerate(cnf_dis):
                diseases.append(it)

    diseases=sorted(set(diseases))

    print("Are you experiencing any ")
    for item in diseases:
        recurse(0, 1,item,feature_names,tree)

    predected_diseases=sorted(set(predected_diseases))

    for item in predected_diseases :
        print("You may have ", item)
        print(description_list[item])
        precution_list.append(precautionDictionary[item])

    precutions=[]

    print("Take following measures : ")
    for j in precution_list:
        for i in j:
            precutions.append(i)

    precutions=sorted(set(precutions))

    for precution in precutions:
        print("-",precution)

    print("Your symtoms : ")
    for d in diseases:
        print(d)


init()
tree_to_code(clf,cols)

inputs = input("")
tokens = nltk.word_tokenize(inputs)
tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in rejected]
for token in tokens:
    isGoodByed,goodbye_response = generate_goodbye_response(token)
    isThanksed,thanks_response = generate_thanks_response(token)
    if (isGoodByed):
        print(goodbye_response)
    if (isThanksed):
        print(thanks_response)


