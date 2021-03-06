#!/usr/bin/python3

#import sys # this library was used in debugging purpose
import numpy as np
from datetime import datetime
from random import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from statistics import stdev

def tn(y_true, y_pred):
    calculated_confusion_matrix = confusion_matrix(y_true, y_pred)[0, 0]
    #print (calculated_confusion_matrix) # Written for debugging purpose
    return calculated_confusion_matrix

def fp(y_true, y_pred):
    calculated_confusion_matrix = confusion_matrix(y_true, y_pred)[0, 1]
    #print (calculated_confusion_matrix) # Written for debugging purpose
    return calculated_confusion_matrix

def fn(y_true, y_pred):
    calculated_confusion_matrix = confusion_matrix(y_true, y_pred)[1, 0]
    #print (calculated_confusion_matrix) # Written for debugging purpose
    return calculated_confusion_matrix

def tp(y_true, y_pred):
    calculated_confusion_matrix = confusion_matrix(y_true, y_pred)[1, 1]
    #print (calculated_confusion_matrix) # Written for debugging purpose
    return calculated_confusion_matrix

# load data
def load_data(dataset_path):
    data = []
    file_read = open(dataset_path, "r")
    for line in file_read:
        data_row = line.split(",")
        data_row[-1] = data_row[-1][0]
        data.append(data_row)

    #below lines are for debugging purpose
    '''
    t = 0
    for row in data:
        print(row)
        t += 1
        if (t == 100):
            break
    sys.exit()
    '''

    #shuffling the data
    shuffle(data)

    # split features by data and labels
    X = []
    y = []
    for row in data:
        str_to_float = list(map(float, row))
        X.append(str_to_float[:57])
        y.append(str_to_float[-1])

    # return X(data) and y(labels, 0 or 1)
    return X, y

def display_scores(clf_data):
    t1 = 0
    for i in clf_data:

        if (t1 > 0):
            print("-" * 60)
        t1 += 1

        t2 = 0
        temp_string = ""
        for j in i:
            temp_string += "\t"

            if(t2 > 1 and t1 != 1):
                temp_string += "\t"
            t2 += 1

            temp_string += str(j)

        print (temp_string)

def display_table_format(table_data):
    t1 = 0
    for i in table_data:

        if (t1 > 0):
            print("-" * 45)
        t1 += 1

        temp_string = ""
        for j in i:
            temp_string += "\t"
            temp_string += str(j)

        print (temp_string)

def svm_SVC(X, y):
    clf = SVC(kernel="sigmoid", C=1)
    scoring = [["SVM", "ACCURACY", "F-measure", "TIME"]]

    temp_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    scores = [round(x, 4) for x in temp_scores]

    temp_time = cross_validate(clf, X, y, cv=10)['fit_time']
    time = [round(x, 4) for x in temp_time]

    temp_recall = cross_val_score(clf, X, y, cv=10, scoring='recall_macro')
    recall = [round(x, 4) for x in temp_recall]

    temp_precision = cross_val_score(clf, X, y, cv=10, scoring='precision_macro')
    precision = [round(x, 4) for x in temp_precision]

    temp_fmeasure = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
    fmeasure = [round(x, 4) for x in temp_fmeasure]

    for i in range(10):
        scoring.append([i+1, scores[i], fmeasure[i], time[i]])

    display_scores(scoring)
    print("\n")
    return scoring

def KNN(X, y):
    clf = KNeighborsClassifier()
    scoring = [["KNN", "ACCURACY", "F-measure", "TIME"]]

    temp_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    scores = [round(x, 4) for x in temp_scores]

    temp_time = cross_validate(clf, X, y, cv=10)['fit_time']
    time = [round(x, 4) for x in temp_time]

    temp_recall = cross_val_score(clf, X, y, cv=10, scoring='recall_macro')
    recall = [round(x, 4) for x in temp_recall]

    temp_precision = cross_val_score(clf, X, y, cv=10, scoring='precision_macro')
    precision = [round(x, 4) for x in temp_precision]

    temp_fmeasure = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
    fmeasure = [round(x, 4) for x in temp_fmeasure]

    for i in range(10):
        scoring.append([i + 1, scores[i], fmeasure[i], time[i]])

    display_scores(scoring)
    print("\n")
    return scoring

def Dtree(X, y):
    clf = tree.DecisionTreeClassifier()
    scoring = [["DTree", "ACCURACY", "F-measure", "TIME"]]

    temp_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    scores = [round(x, 4) for x in temp_scores]

    temp_time = cross_validate(clf, X, y, cv=10)['fit_time']
    time = [round(x, 4) for x in temp_time]

    temp_recall = cross_val_score(clf, X, y, cv=10, scoring='recall_macro')
    recall = [round(x, 4) for x in temp_recall]

    temp_precision = cross_val_score(clf, X, y, cv=10, scoring='precision_macro')
    precision = [round(x, 4) for x in temp_precision]

    temp_fmeasure = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
    fmeasure = [round(x, 4) for x in temp_fmeasure]

    for i in range(10):
        scoring.append([i + 1, scores[i], fmeasure[i], time[i]])

    display_scores(scoring)
    print("\n")
    return scoring






def get_ranks(val):
    ranks = []
    ranks.append(val[0])
    length = len(val)
    sorted_list = []
    
    for i in range(1,length):    
        sorted_list.append(val[i])
    sorted_list = sorted(sorted_list)
    #print(sorted_list)
        
        
    for i in range(1,length):
        
        indx = sorted_list.index(val[i])
        curr_rank = indx + 1        
        ranks.append(curr_rank)
    
    return ranks        


def friedman_test(arr):
    
    ranks = []
    
    for i in range(11):
        if(i == 0):
            ranks.append(arr[i])
            continue
            
        ranks.append(get_ranks(arr[i]))
    
    svm_rank_sum = 0
    knn_rank_sum = 0
    dtr_rank_sum = 0
    
    for i in range(10):
        svm_rank_sum += ranks[i+1][1]
        knn_rank_sum += ranks[i+1][2]
        dtr_rank_sum += ranks[i+1][3]
        
    svm_rank_avg = svm_rank_sum/10
    knn_rank_avg = knn_rank_sum/10
    dtr_rank_avg = dtr_rank_sum/10
        
    ranks.append(['sum',   svm_rank_sum,    knn_rank_sum,  dtr_rank_sum])
    ranks.append(['avg',   round(svm_rank_avg, 1),    round(knn_rank_avg, 1),  round(dtr_rank_avg, 1)])
    
    display_table_format(ranks)
    print("\n")
    
    n = 10
    k = 3
    
    friedman_stat = ( 12 / (n * k * (k+1)) ) * (svm_rank_sum**2 + knn_rank_sum**2 + dtr_rank_sum**2)
    friedman_stat -= 3 * n * (k+1)
    
    print (f"Friedman statistic is = {friedman_stat}")
    
    level = 7.8
    print (f"The critical value for k = 3 and n = 10 at the α = 0.05 level is = {level}\n")
    
    post_hoc = 0
    if (friedman_stat > level):
        print ("There are significant differences.")
        print ("Null hypothesis is rejected.")
        post_hoc = 1
    else:
        print ("There are no significant differences.")
        print ("Null hypothesis cannot be rejected.")
        #print ("Nemenyi test will not be performed.")
    
    if(post_hoc == 1):
        #print("Nemenyi test will be performed.")
        cd = 2.343 * ( (k*(k+1)) / (6*n) )**0.5
        print(f"\nCritical difference is = {cd}")
        
        if (abs(svm_rank_avg - knn_rank_avg) > cd):
            print("Significant difference was found between SVM and K-Nearest Neighbour.")
        if (abs(knn_rank_avg - dtr_rank_avg) > cd):
            print("Significant difference was found between K-Nearest Neighbour and Decision Tree.")
        if (abs(svm_rank_avg - dtr_rank_avg) > cd):
            print("Significant difference was found between SVM and Decision Tree.")
           
    print ("\n\n\n")

    
    
    
    
    


start_time = datetime.utcnow()
temp_time_counter = 0
time_counter = 0
database_file = "spambase.data"

print (f"Loading database form '{database_file}'.")
X, y = load_data(database_file)

time_counter = datetime.utcnow() - start_time
print (f"Time taken for loading database: {time_counter}\n\n")

temp_time_counter = datetime.utcnow()
svmsvc = svm_SVC(X, y)
time_counter = datetime.utcnow() - temp_time_counter
print (f"Execution time for computing SVM: {time_counter}\n\n\n")

temp_time_counter = datetime.utcnow()
knn = KNN(X, y)
time_counter = datetime.utcnow() - temp_time_counter
print (f"Execution time for computing KNN: {time_counter}\n\n\n")

temp_time_counter = datetime.utcnow()
decisionTree = Dtree(X, y)
time_counter = datetime.utcnow() - temp_time_counter
print (f"Execution time for computing DTREE: {time_counter}\n\n\n")









final_list_acc = [["Fold", "SVM", "KNN", "DTREE"]]
svmsvc_acc_values = []
knn_acc_values = []
dTree_acc_values = []


final_list_F = [["Fold", "SVM", "KNN", "DTREE"]]
svmsvc_F_values = []
knn_F_values = []
dTree_F_values = []


final_list_time = [["Fold", "SVM", "KNN", "DTREE"]]
svmsvc_time_values = []
knn_time_values = []
dTree_time_values = []


for i in range(10):
    #print(f"{i+1} {svmsvc[i+1][1]} {knn[i+1][1]} {decisionTree[i+1][1]}")
    
    svmsvc_acc_values.append(svmsvc[i+1][1])
    knn_acc_values.append(knn[i+1][1])
    dTree_acc_values.append(decisionTree[i+1][1])    
    final_list_acc.append([i+1, svmsvc[i+1][1], knn[i+1][1], decisionTree[i+1][1]])
    
    svmsvc_F_values.append(svmsvc[i+1][2])
    knn_F_values.append(knn[i+1][2])
    dTree_F_values.append(decisionTree[i+1][2])    
    final_list_F.append([i+1, svmsvc[i+1][2], knn[i+1][2], decisionTree[i+1][2]])
    
    svmsvc_time_values.append(svmsvc[i+1][3])
    knn_time_values.append(knn[i+1][3])
    dTree_time_values.append(decisionTree[i+1][3])    
    final_list_time.append([i+1, svmsvc[i+1][3], knn[i+1][3], decisionTree[i+1][3]])

final_list_acc.append(["avg", round(np.mean(svmsvc_acc_values), 4), round(np.mean(knn_acc_values), 4), round(np.mean(dTree_acc_values), 4)])
final_list_acc.append(["stdev", round(stdev(svmsvc_acc_values), 4), round(stdev(knn_acc_values), 4), round(stdev(dTree_acc_values), 4)])

final_list_F.append(["avg", round(np.mean(svmsvc_F_values), 4), round(np.mean(knn_F_values), 4), round(np.mean(dTree_F_values), 4)])
final_list_F.append(["stdev", round(stdev(svmsvc_F_values), 4), round(stdev(knn_F_values), 4), round(stdev(dTree_F_values), 4)])

final_list_time.append(["avg", round(np.mean(svmsvc_time_values), 4), round(np.mean(knn_time_values), 4), round(np.mean(dTree_time_values), 4)])
final_list_time.append(["stdev", round(stdev(svmsvc_time_values), 4), round(stdev(knn_time_values), 4), round(stdev(dTree_time_values), 4)])










print ("\n")
print ("Accuracy values:\n")
display_table_format(final_list_acc)
print ("\n")

print("Friedman test for Accuracy:\n")
friedman_test(final_list_acc)



print ("\n")
print ("F-measure values:\n")
display_table_format(final_list_F)
print ("\n")

print("Friedman test for F-measure:\n")
friedman_test(final_list_F)



print ("\n")
print ("Time values:\n")
display_table_format(final_list_time)
print ("\n")

print("Friedman test for Time:\n")
friedman_test(final_list_time)



time_counter = datetime.utcnow() - start_time
print (f"Total execution time: {time_counter}\n")
