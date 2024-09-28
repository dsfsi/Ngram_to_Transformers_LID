import matplotlib
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import string

from collections import defaultdict
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pickle as pkl
# from yellowbrick.classifier import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.probability import FreqDist
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
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

def read_and_init(train_data_root_path):
            #Train, Dev, Test
            # Step 4: Split the data
            # Check if the current item is a file
            data_tr = []
            data_d = []
            data_te = []
            data_dix_train = {}
            data_dix_test = {}
            data_dix_dev = {}
            # data_train_dev_test = {}
            for item in os.listdir(train_data_root_path):
                    # Construct the full path to the current item
                    item_path = os.path.join(train_data_root_path , item)
                    # Check if the current item is a directory
                    if os.path.isdir(item_path):
                        print("-------------------------------------------------------------")
                        print("Collecting data For language: ", item_path.split("/")[-1])

                        # Loop through the contents of the subfolder
                        for sub_item in os.listdir(item_path):
                            # Construct the full path to the sub-item
                            sub_item_path = os.path.join(item_path, sub_item)

                            # Check if the sub-item is a file or directory
                            if os.path.isdir(sub_item_path):
                                language = item_path.split("/")[-1]
                                files_manager = {}
                                for file_name in os.listdir(sub_item_path):
                                        file_path = os.path.join(sub_item_path, file_name)
                                        #print(file_path)
                                        if 'train' in file_name:
                                                files_manager['train_file'] = file_path
                                                data_tr.append(pd.read_csv(files_manager['train_file']))
                                                data_dix_train[language] = list(pd.read_csv(files_manager['train_file'])['sentence'])
                                        elif 'dev' in file_name:
                                                files_manager['dev_file'] = file_path
                                                data_d.append(pd.read_csv(files_manager['dev_file']))
                                        elif 'test' in file_name:
                                                files_manager['test_file'] = file_path  
                                                data_te.append(pd.read_csv(files_manager['test_file']))
                                                data_dix_test[language] = list(pd.read_csv(files_manager['test_file'])['sentence'])
                                        else:
                                                print("This file is unknow") 
                            else:
                                # its a file
                                files_manager = {}
                                if 'train' in sub_item_path:
                                        files_manager['train_file'] = sub_item_path
                                        data_tr.append(pd.read_csv(files_manager['train_file']))
                                        data_dix_train[item_path.split("/")[-1]] = list(pd.read_csv(files_manager['train_file'])['sentence'])
                                elif 'dev' in sub_item_path:
                                        files_manager['dev_file'] = sub_item_path
                                        data_d.append(pd.read_csv(files_manager['dev_file']))
                                        data_dix_dev[item_path.split("/")[-1]] = list(pd.read_csv(files_manager['dev_file'])['sentence'])

                                elif 'test' in sub_item_path:
                                            files_manager['test_file'] = sub_item_path  
                                            data_te.append(pd.read_csv(files_manager['test_file']))
                                            data_dix_test[item_path.split("/")[-1]] = list(pd.read_csv(files_manager['test_file'])['sentence'])
                                else:
                                    print("This file is unknow")
            df_train  =    pd.concat(data_tr, ignore_index = True, sort=False)
            df_test  =    pd.concat(data_te, ignore_index = True, sort=False)
            return  df_test 


root_path = "../Csv_Train_test_split/"
df_test_data = read_and_init(root_path)
N = df_test_data.shape[0]
clean_df  = pd.DataFrame(columns=['sentence','label'])
texts = list(df_test_data['sentence'].values)
true_labels  = list(df_test_data['label'].values)
for i in range(N):
         text_line = clean_text(texts[i])
         if text_line != '':
                    split_s = text_line.split()
                    if len(split_s) >=3 and len(split_s) <= 50:
                                   clean_df.loc[len(clean_df)] = [texts[i], true_labels[i]]

clean_df = clean_df.reset_index(drop=True)

import os, sys
import logging
from afrolid.main import classifier
     

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    force=True, # Resets any previous configuration
)
logger = logging.getLogger("afrolid")

     

cl = classifier(logger, model_path="/home/neo/afrolid_model/")
     
def get_afrolid_prediction(text):
  predictions = cl.classify(text, max_outputs=1)
  for lang in predictions:
    return lang, predictions[lang]['score'], predictions[lang]['name'], predictions[lang]['script']
  
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import argparse
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        args = parser.parse_args()
        df_test_data['predict_iso'], df_test_data['predict_score'], df_test_data['predict_name'], df_test_data['predict_script'] = zip(*df_test_data['sentence'].progress_apply(get_afrolid_prediction))
        # df_test_data.head()

        df_test_data.to_csv('../AfroLID_Output/AfroLID_Output_CSV_' + str(args.seed)+'.csv', index=False)

        clean_df['predict_iso'], clean_df['predict_score'], clean_df['predict_name'], clean_df['predict_script'] = zip(*clean_df['sentence'].progress_apply(get_afrolid_prediction))
        # df_test_data.head()

        clean_df.to_csv('../AfroLID_Output/AfroLID_Output_CSV_Clean_Text_' + str(args.seed)+'.csv', index=False)
