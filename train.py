from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import os
import numpy as np
import re
import cv2
from glob import glob
import matplotlib.pyplot as plt
import random

# Path = '/content/drive/My Drive/Colab Notebooks/C2_TrainDev_Toy'
Path = './C2_TrainDev_Toy'

def load_mango_csv(csv_path):
    path = []
    box = []
    label = []
    subdir = csv_path.split('/')[-1].split('.')[0].capitalize()
    with open(csv_path, 'r', encoding='utf8') as f:
        for line in f:
            clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
            curr_img_path = f'{Path}/{subdir}/{clean_line[0]}'
            curr_info = np.array(clean_line[1:]).reshape(-1, 5)
            curr_box = curr_info[:, :-1].astype('float16').tolist()
            curr_label = curr_info[:, -1].tolist()
            path.append(curr_img_path)
            box.append(curr_box)
            label.append(curr_label)

    return path, box, label

def load_data():
    if os.path.isfile(f'{Path}/X_train.npy') and os.path.isfile(f'{Path}/y_train.npy') and os.path.isfile(f'{Path}/X_dev.npy') and os.path.isfile(f'{Path}/y_dev.npy'):
        X_train_total = np.load(f'{Path}/X_train.npy')
        X_dev_total = np.load(f'{Path}/X_dev.npy')
        label_train_total = np.load(f'{Path}/y_train.npy')
        label_dev_total = np.load(f'{Path}/y_dev.npy')
    else:
        X_train_total, label_train_total = load_image(dataset='train')
        X_dev_total, label_dev_total = load_image(dataset='dev')
        np.save(f'{Path}/X_train', X_train_total)
        np.save(f'{Path}/y_train', label_train_total)
        np.save(f'{Path}/X_dev', X_dev_total)
        np.save(f'{Path}/y_dev', label_dev_total)

    return X_train_total, X_dev_total, label_train_total, label_dev_total

def load_dev_image_TSV():
    X = []
    img_name = []
    csv_path = f'{Path}/dev.csv'
    with open(csv_path, 'r', encoding='utf8') as f:
        for line in f:
            clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
            curr_img_path = f'{Path}/Dev/{clean_line[0]}'
            try:
                img = cv2.cvtColor(cv2.imread(curr_img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                X.append(img)
                img_name.append(clean_line[0])
            except:
                continue

    _X = extract_features(np.array(X))
    img_name = np.array(img_name)

    return _X, img_name

def load_image(dataset):
    defect_map = {
        '不良-乳汁吸附': 0,
        '不良-機械傷害': 1,
        '不良-炭疽病': 2,
        '不良-著色不佳': 3,
        '不良-黑斑病': 4
    }

    path, box, label = load_mango_csv(csv_path=f'{Path}/{dataset}.csv')
    batch_size = 5000
    for batch in range(len(path) // batch_size + 1):
        X = []
        y_label = []
        print(batch)
        for i in range(batch_size * batch, batch_size * (batch+1)):
            try:
                if i == len(path):
                    return _X, _y_label
                img = cv2.cvtColor(cv2.imread(path[i]), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                X.append(img)
                defect = [0,0,0,0,0]
                for j in range(len(label[i])):
                    defect_idx = defect_map[label[i][j]]
                    defect[defect_idx] = 1
                y_label.append(defect)
            except:
                continue

        if batch == 0:
            _X = extract_features(np.array(X))
            _y_label = np.array(y_label)
        else:
            X = extract_features(np.array(X))
            _X = np.vstack((_X, X))
            _y_label = np.vstack((_y_label, np.array(y_label)))

    return _X, _y_label

def extract_features(X):
    vgg16_base = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3), classifier_activation=None)
    resnet50_base = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3), classifier_activation=None)
    X = preprocess_input(X)
    vgg16_features = vgg16_base.predict(X)
    resnet50_features = resnet50_base.predict(X)
    features = np.hstack((vgg16_features, resnet50_features))

    return features

def get_defect_balance_data(dataset, defect, X_total, label_total):
    defect_idx = [ i for i in range(len(label_total)) if label_total[i][defect] == 1]
    defect_length = len(defect_idx)
    non_defect_idx = np.delete(np.arange(len(label_total)), defect_idx)
    non_defect_length = len(non_defect_idx)
    if dataset == 'train':
        if defect_length < non_defect_length:
            non_defect_idx = np.random.choice(non_defect_idx, defect_length, replace=False)
            y = np.hstack((np.ones(shape=(defect_length, )), np.zeros(shape=(defect_length, ))))
        else:
            defect_idx = np.random.choice(defect_idx, non_defect_length, replace=False)
            defect_idx = list(defect_idx)
            y = np.hstack((np.ones(shape=(non_defect_length, )), np.zeros(shape=(non_defect_length, ))))
    else:
        y = np.hstack((np.ones(shape=(defect_length, )), np.zeros(shape=(non_defect_length, ))))

    defect_idx.extend(list(non_defect_idx)) # all index
    X = X_total[defect_idx]

    return X, y

def VGG16_ANOVA_SVM(SAVE_TSV, X_train, y_train, X_dev, y_dev, anova_percentile, complexity):

    print('train linear svm model...')
    svm = LinearSVC(random_state=42, C=complexity, class_weight='balanced')
    clf = Pipeline([('scaler', MinMaxScaler()),
                    ('anova', SelectPercentile(chi2)),
                    # ('scaler', StandardScaler()),
                    ('svc', svm)])

    clf.set_params(anova__percentile=anova_percentile)
    clf.fit(X_train, y_train)
    pred_y_dev = clf.predict(X_dev)

    if SAVE_TSV:
        return pred_y_dev

    print('evaluating dev data...')
    cm = confusion_matrix(y_dev, pred_y_dev)
    acc = accuracy_score(y_dev, pred_y_dev)
    f1 = f1_score(y_dev, pred_y_dev)
    p = precision_score(y_dev, pred_y_dev)
    r = recall_score(y_dev, pred_y_dev)

    print(cm, acc, f1, p, r)
    print('---------------------------------------------------------')

    return p, r

if __name__ == '__main__':

    SAVE_TSV = False
    defect_P_C_map = {
        0: (50, 1.0), # 50 1
        1: (10, 1),  # 10 0.1
        2: (20, 0.1), # 20 0.1
        3: (100, 1),  # 10 0.1
        4: (20, 1.0)  # 20 1
    }


    '''
        Load all images after feature extraction(X) and all defects(label)
        Both:
            X_train_total: (len(train), 2000) -> 2000 features
            label_train_total: (len(train), 5) -> 5 defects ex:[0,0,0,0,0]
            X_dev_total: (len(dev), 2000)
        TSV:
            img_name: (len(dev), 1) -> 1 image's name ex:01389.jpg
        without TSV:
            label_dev_total: (len(dev), 5)
    '''
    if SAVE_TSV:
        X_train_total, label_train_total = load_image(dataset='train')
        X_dev_total, img_name = load_dev_image_TSV()
        img_name = np.expand_dims(img_name, axis=1)
    else:
        X_train_total, X_dev_total, label_train_total, label_dev_total = load_data()


    '''
        Get each defect data, balance train data and predict
        TSV:
            get predicts change to `True` or `False`
        without TSV:
            calculate precision and recall
    '''
    precision = 0
    recall = 0
    for defect in range(len(defect_P_C_map)):
        X_train, y_train = get_defect_balance_data('train', defect, X_train_total, label_train_total)

        if SAVE_TSV:
            preds = VGG16_ANOVA_SVM(SAVE_TSV, X_train, y_train, X_dev_total, img_name, anova_percentile=defect_P_C_map[defect][0], complexity=defect_P_C_map[defect][1])
            preds = list(preds)
            _preds = []
            for i in range(len(preds)):
                if preds[i] == 1.0:
                    _preds.append("True")
                else:
                    _preds.append("False")
            _preds = np.expand_dims(np.array(_preds), axis=1)
            img_name = np.hstack((img_name, _preds))
        else:
            X_dev, y_dev = get_defect_balance_data('dev', defect, X_dev_total, label_dev_total)
            p, r = VGG16_ANOVA_SVM(SAVE_TSV, X_train, y_train, X_dev, y_dev, anova_percentile=defect_P_C_map[defect][0], complexity=defect_P_C_map[defect][1])
            precision += p
            recall += r


    '''
        TSV:
            save tsv file
        without TSV:
            calculate f1 score
    '''
    if SAVE_TSV:
        results = img_name
        np.savetxt("E24066022_predict.tsv", results, delimiter="\t", fmt='%s')
    else:
        precision_ma = precision / 5
        recall_ma = recall / 5
        F1_ma = 2 * precision_ma * recall_ma / (precision_ma + recall_ma)
        print(F1_ma)


