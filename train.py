#coding='utf-8'
import os
from PIL import Image
from feature import NPDFeature
import numpy as np
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data():
    face_path = u'C:/Users/47864/Desktop/Data/datasets/original/face'
    nonface_path = u'C:/Users/47864/Desktop/Data/datasets/original/nonface'
    face_image = os.listdir(face_path)
    nonface_image = os.listdir(nonface_path)
    num_face_image = len(face_image)
    num_nonface_image = len(nonface_image)
    
    dataset = []
    for i in range(num_face_image):
        img = Image.open(face_path + '/' + face_image[i])
        img = img.convert('L')
        img = img.resize((24,24),Image.ANTIALIAS)
        img = NPDFeature(np.array(img))
        dataset.append(np.concatenate((img.extract(),np.array([1]))))

    
    for i in range(num_nonface_image):
        img = Image.open(nonface_path + '/'+nonface_image[i])
        img = img.convert('L')
        img = img.resize((24,24),Image.ANTIALIAS)
        img = NPDFeature(np.array(img))
        dataset.append(np.concatenate((img.extract(),np.array([-1]))))
        
    return dataset
    
if __name__ == "__main__":
    # write your code here
    #dataset = load_data()
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3),6)
    #classifier.save(dataset,'C:/Users/47864/Desktop/Data/data.txt')
    dataset = classifier.load('C:/Users/47864/Desktop/Data/data.txt')
    dataset = np.array(dataset)
    X = dataset[:,:-1]
    y = dataset[:,-1:]
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2)
    #print(len(X),len(X_train))
    #print(X_train[0])
    
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_val)
    #print(y_pred,y_val)
    accuracy_rate = np.sum(y_pred.T == y_val.T[0])/float(len(y_pred))
    target_names = ['nonface','face']
    print("1")
    print(y_val.T[0])
    print("1")
    print(accuracy_rate)
    with open('C:/Users/47864/Desktop/Data/report.txt','w') as report:
        report.write(classification_report(y_val.T[0],y_pred.T,target_names = target_names))
