import pickle
from collections import Counter
import math
import re
import os
import string
import argparse

# https://github.com/dinkarjuyal/language-identification/blob/master/lang%2Bidentify.ipynb
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np

from collections import defaultdict
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.probability import FreqDist
from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import joblib
import pickle as pkl

def preprocess_text( text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers (retain only alphabets and spaces)
    # text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

def vectorize_text(train_texts, test_texts, ngram_range, is_word=False):
    if is_word == False:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, preprocessor=preprocess_text)
            X_train = vectorizer.fit_transform(train_texts)
            # X_val = vectorizer.transform(val_texts)
            X_test = vectorizer.transform(test_texts)
    else:
            vectorizer = TfidfVectorizer(analyzer='word', preprocessor=preprocess_text)
            X_train = vectorizer.fit_transform(train_texts)
            # X_val = vectorizer.transform(val_texts)
            X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer


def clean_text( text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers (retain only alphabets and spaces)
    # text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert text to lowercase
    text = text.lower()
    
    return text



def preprocess_eval(text):
        text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
        text = str(text.strip())
        # text  = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = " ".join([st for st in text.split(" ") if st != ''])
        return text
          

eval_data_root_path = "../Train_test_split/"
train_data_root_path = "../../../../../ext_data/neo/Chunks_Vuk_p_NCHLT_train_only/"

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
         
        args = parser.parse_args()
        train_data_in_seed = {}
        lang_list = []
        # print(os.listdir(train_data_root_path))
        for item in os.listdir(train_data_root_path):
            # Construct the full path to the current item
            item_path = os.path.join(train_data_root_path , item)
            # Check if the current item is a directory
            if os.path.isdir(item_path):
                print("-------------------------------------------------------------")
                print("Generating N-gram For language: ", item_path.split("/")[-1])
                language = item_path.split("/")[-1]
                lang_list.append(language)
                # Loop through the contents of the subfolder
                for sub_item in os.listdir(item_path):
                    # Construct the full path to the sub-item
                    sub_item_path = os.path.join(item_path, sub_item)

                    # Check if the sub-item is a file or directory
                    if os.path.isdir(sub_item_path):
                        lang = item_path.split("/")[-1]
                        files_manager = {}
                        for file_name in os.listdir(sub_item_path):
                                sub_sub_path = os.path.join(sub_item_path, file_name)
                                # print(int(sub_item))
                                if os.path.isdir(sub_sub_path):
                                          for sub_sub_file in sub_sub_path:
                                                         sub_sub_file_p = os.path.join(sub_item_path, sub_sub_file)
                                                         if os.path.isfile(sub_sub_file_p):
                                                                    if int(sub_sub_path.split("/")[-1]) == int(args.seed):
                                                                                # print(sub_sub_file_p)
                                                                                corpus   = open(sub_sub_file_p, 'r')
                                                                                corpus = corpus.readlines()
                                                                                train_data_in_seed[lang] = corpus
                                                                                            
                                                         else:
                                                                  print("Item is folder")  

                                else:
                                        if int(sub_item) == int(args.seed):
                                                    # print(sub_sub_file_p)
                                                    corpus   = open(sub_sub_path, 'r')
                                                    corpus = corpus.readlines()
                                                    train_data_in_seed[lang] = corpus


        train_preprocessed = {k: [clean_text(sentence) for sentence in v] for k, v in train_data_in_seed.items()}
        sentences_train, y_train =[], []
        for k, v in train_preprocessed.items():
            for sentence in v:
                sentences_train.append(sentence)
                y_train.append(k)
        
        test_data_in_seed = {}
        val_data_in_seed  = {}
        for item in os.listdir(eval_data_root_path):
            # Construct the full path to the current item
            item_path = os.path.join(eval_data_root_path, item)
            # Check if the current item is a directory
            if os.path.isdir(item_path):
                # Loop through the contents of the subfolder
                for sub_item in os.listdir(item_path):
                    # Construct the full path to the sub-item
                    sub_item_path = os.path.join(item_path, sub_item)

                    # Check if the sub-item is a file or directory
                    if os.path.isdir(sub_item_path):
                        language_test = item_path.split("/")[-1]
                        files_manager = {}
                        for file_name in os.listdir(sub_item_path):
                                file_path = os.path.join(sub_item_path, file_name)

                                # Check if the current item is a file
                                if os.path.isfile(file_path):
                                            #print("File:", file_path)
                                            if 'train' in file_name:
                                                    files_manager['train_file'] = file_path
                                            elif 'dev' in file_name:
                                                    files_manager['dev_file'] = file_path
                                            elif 'test' in file_name:
                                                    files_manager['test_file'] = file_path  
                                            else:
                                                    print("This file is unknow")                              

                        # Test
                        if  files_manager['test_file']:    
                                    corpus   = open(files_manager['test_file'], "r")
                                    corpus = corpus.readlines()
                                    test_data_in_seed[language_test] = corpus

                        # Dev 
                        # if  files_manager['dev_file']:    
                        #             corpus   = open(files_manager['dev_file'], "r")
                        #             corpus = corpus.readlines()
                        #             val_data_in_seed[language_test] = corpus            
        
        test_preprocessed  = {k: [clean_text(sentence) for sentence in v] for k, v in test_data_in_seed.items()}
        sentences_test, y_test =[], []
        for k, v in test_preprocessed.items():
            for sentence in v:
                sentences_test.append(sentence)
                y_test.append(k)

        def evaluation_plots(y_test, y_pred, languages, fname, show = True, verbose = True):
            y_test = [languages[item] for item in y_test]
            y_pred = [languages[item] for item in y_pred]
            confusion_matr = confusion_matrix(y_test,y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1_sco = f1_score(y_test, y_pred, average = "weighted")
            precision = precision_score(y_test, y_pred,average = "weighted")
            recall = recall_score(y_test, y_pred, average = "weighted")

            if fname:
                df_cm = pd.DataFrame(confusion_matr, index = [i for i in list(languages.keys())],
                        columns = [i for i in list(languages.keys())])
                sn.set(font_scale=1)#for label size
                plt.figure(figsize=(10, 7))
                plt.tight_layout()
                sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, fmt='g')# font
                plt.title("Accuracy = " + str(accuracy),size = 18)
                plt.ylabel("Actual Label")
                plt.xlabel("Predicted Label")
                plt.savefig(fname + '/conf_matrix.pdf', dpi=300)
                plt.savefig(fname + '/conf_matrix.jpeg', dpi=300)
                plt.clf()
                # plt.show()

            metrics = {
                    "accuracy": accuracy,
                    "f1_score": f1_sco,
                    "precision": precision,
                    "recall": recall}
            # wrong_ixd = [i for i,pred in enumerate(y_pred) if not pred == y_test[i]]
            # wrong_pred = np.array(y_pred)[wrong_ixd]

            # metrics["wrong_ixd"] = wrong_ixd
            # metrics["wrong_pred"] = wrong_pred
            # if verbose:
            #     print(metrics)
            #     print("wrong_ixd",wrong_ixd)
            #     print("wrong_pred",wrong_pred)
            output_test_results_file = fname + "/test_scores.txt"
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics.keys()):
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

            return metrics


        # Train and evaluate a model
        def train_and_evaluate_model(model_path_directory, model, X_train, y_train, X_test, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            languages = { lang: i for i, lang in enumerate(['nso', 'tsn', 'sot','ven', 'af', 'tso', 'ssw', 'zul', 'xho', 'nbl', 'eng'])}
            met = evaluation_plots(y_test, y_pred, languages, model_path_directory)

            
            return classification_report(y_test, y_pred)

        # Main pipeline function
        def main_pipeline(sentences_train, y_train, sentences_test, y_test , is_word, ngram_range):

            X_train_vec, X_test_vec, vectorizer = vectorize_text(sentences_train, sentences_test, ngram_range, is_word)
            
            # Initialize models
            models = {
                'KNN': KNeighborsClassifier(),
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Naive Bayes': MultinomialNB(),
                'SVM': SVC()
            }
            
            # Train and evaluate each model
            results = {}
            for model_name, model in models.items():
                dir_results_ablation = "../ML_Ablation_Results/" + str(args.seed)
                if not os.path.isdir(dir_results_ablation):
                    os.mkdir(dir_results_ablation)
                model_path_directory = dir_results_ablation + "/" + model_name + '/' + "".join(map(str,ngram_range))   
                os.makedirs(model_path_directory)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                results[model_name] = train_and_evaluate_model(model_path_directory, model, X_train_vec, y_train, X_test_vec, y_test)
            
            # Print evaluation results
            for model_name, result in results.items():
                print(f"Results for {model_name} with n-grams {ngram_range}:\n{result}\n")                                    

        print("Training and evaluating with ungrams:")
        main_pipeline(sentences_train,y_train, sentences_test, y_test, False, ngram_range=(1, 1))

        print("Training and evaluating with bigrams:")
        main_pipeline(sentences_train,y_train, sentences_test, y_test, False, ngram_range=(2, 2))

        print("Training and evaluating with trigrams:")
        main_pipeline(sentences_train,y_train, sentences_test, y_test, False ,ngram_range=(3, 3))

        print("Training and evaluating with quadgrams:")
        main_pipeline(sentences_train,y_train, sentences_test, y_test, False, ngram_range=(4, 4))   

        print("Training and evaluating with N-grams-comb:")
        main_pipeline(sentences_train, y_train, sentences_test, y_test, False, ngram_range=(2, 4))  

        print("Training and evaluating with Word TFIDF")
        main_pipeline(sentences_train, y_train, sentences_test, y_test, True, ngram_range=('word', 'word')) 


                                                                                                                                       