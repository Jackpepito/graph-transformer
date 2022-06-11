import numpy as np
import os
import csv
import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split

class Writer(object):
    #data root: folder dataset
    #site: dataset name
    #patches_folder: patches folder in data root
    #feature_extractor_path: dir for all_patches.csv
    #metadata_name= file csv in site folder with dataset metadata


    def __init__(self, data_root, site, patches_folder, feature_extractor_path, metadata_name):
        self.data_root = data_root
        self.site = os.path.join(self.data_root, self.site)
        self.patches_path = os.path.join(self.root, patches_folder)
        self.feature_extractor_path = feature_extractor_path
        self.metadata = os.path.join(self.site, self.metadata_name)

    #tutti i percorsi delle patch vengono scritti su all_patches.csv
    def write_patch_path(self):
        path_list = []

        for path, subdirs, files in os.walk(self.patches_path):
            for name in files:
                if not subdirs:
                p = []
                p.append(os.path.join(path, name))
                path_list.append(p)

        # opening the csv file in 'a+' mode
        file = open(os.path.join(self.feature_extractor_path, "all_patches.csv", 'a+', newline =''))

        # writing the data into the file
        with file:   
            write = csv.writer(file)
            write.writerows(path_list)

        file.close()

    #scrive file txt che contiene le wsi per train, val, test per il transformer 
    def write_split(self):
        dfs = pd.concat(pd.read_excel(self.metadata, sheet_name=None), ignore_index=True)
        train, test = train_test_split(dfs, test_size=0.2)
        train, val= train_test_split(train, test_size=0.1)
        bags = [train, val, test]
        split = ['train', 'val', 'test']

        i=0
        for bag in bags:
            with open(self.site + split[i] + '.txt', 'w') as f:
                for index, row in bag.iterrows():
                    item=row['Image No.'] + '\\t' + row['Treatment effect'] 
                    f.write("%s\n" % item)
            i+=1

        del dfs, bags
        del train, val, test