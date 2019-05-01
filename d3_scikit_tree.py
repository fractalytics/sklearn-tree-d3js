import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier  

#load data from flat file

df=pd.read_csv("bank.csv",sep=';')
df.dropna(inplace=True)


#set the label colomne 
label_name = 'y'
df.sort([label_name], ascending=[True], inplace=True)
df= df.sort_index(by=[label_name], ascending=[True])

features = (df.drop(label_name,axis=1).columns.values)

is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
boolfeatures= is_number(df.drop(label_name,axis=1).dtypes)

df_dummy = pd.get_dummies(df.drop(label_name,axis=1),prefix_sep='_-_')




def generator_1(clf, features, labels,original_features, node_index=0,side=0):
  
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
        node['size'] = sum( clf.tree_.value[node_index, 0]  )   
        node['side'] = 'left' if side == 'l' else 'right'                      
    else:

        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['pred'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
                                      
        node['side'] = 'left' if side == 'l' else 'right'                              
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        
        if ('_-_' in feature) and (feature not in original_features):
            node['name'] =  '{} = {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] )
            node['type'] = 'categorical'
        else:
            node['name'] = '{} > {}'.format(feature, round(threshold,2) )
            node['type'] = 'numerical'
        
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        
        node['size'] = sum (clf.tree_.value[node_index, 0])
        node['children'] = [generator_1(clf, features, labels, original_features, right_index,'r'),
                            generator_1(clf, features, labels, original_features, left_index,'l')]
                            
        
    return node



def generator_2(clf, features, labels,original_features, node_index=0,side=0,prev_index=0):

    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['pred'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
                                      
        node['side'] = 'left' if side == 'l' else 'right'                              
        feature = features[clf.tree_.feature[prev_index]]
        threshold = clf.tree_.threshold[prev_index]
        
            
        if node_index == 0:
            node["name"] = 'Root >'
        elif ('_-_' in feature) and (feature not in original_features):
            
            node['name'] =  '{} = {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] ) if side == 'r' else '{} != {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] )  
            node['type'] = 'categorical'
        else:
            node['name'] = '{} > {}'.format(feature, round(threshold,2) ) if side == 'r' else '{} <= {}'.format(feature, round(threshold,2) ) 
            node['type'] = 'numerical'
        
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        
        node['size'] = sum (clf.tree_.value[node_index, 0])
           
    else:

        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['pred'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
                                      
        node['side'] = 'left' if side == 'l' else 'right'                              
        feature = features[clf.tree_.feature[prev_index]]
        threshold = clf.tree_.threshold[prev_index]
        
            
        if node_index == 0:
            node["name"] = 'Root >'
        elif ('_-_' in feature) and (feature not in original_features):
            
            node['name'] =  '{} = {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] ) if side == 'r' else '{} != {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] )  
            node['type'] = 'categorical'
        else:
            node['name'] = '{} > {}'.format(feature, round(threshold,2) ) if side == 'r' else '{} <= {}'.format(feature, round(threshold,2) ) 
            node['type'] = 'numerical'
        
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        
        node['size'] = sum (clf.tree_.value[node_index, 0])
        node['children'] = [generator_2(clf, features, labels, original_features, right_index,'r',node_index),
                            generator_2(clf, features, labels, original_features, left_index,'l',node_index)]
                            
        
    return node
    
    


clf = DecisionTreeClassifier(max_depth=5)
clf.fit(df_dummy, df[label_name])

io=generator_1(clf, df_dummy.columns,np.unique(df[label_name]),features)

print(json.dumps(io, indent=4))

with open('structureC1.json', 'w') as outfile:
    json.dump(io, outfile, indent=4)
    
    
io=generator_2(clf, df_dummy.columns,np.unique(df[label_name]),features)

print(json.dumps(io, indent=4))

with open('structureC2.json', 'w') as outfile:
    json.dump(io, outfile, indent=4)