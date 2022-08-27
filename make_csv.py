import torch
import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm


def get_label_df(data_path):

    def range_to_index(age_range):
        if age_range == '18-24':
            return 0
        elif age_range == '25-34':
            return 1
        elif age_range == '35-49':
            return 2
        elif age_range == '50-XX':
            return 3
        return -1

    label_df = pd.DataFrame(columns=["ID", "AGE"])
    with open(os.path.join(data_path, 'truth.txt')) as fp:
        for line in fp:
            data = line.split(":::")
            label_df.loc[len(label_df)] = [data[0], range_to_index(data[2])]
    return label_df


def get_document_df(data_path):
    document_df = pd.DataFrame(columns=['ID', 'document'])
    base_path = os.path.join(data_path, 'data')
    print("xml parsing...")
    for xml_file_name in tqdm(os.listdir(base_path)):
        xml_path = os.path.join(base_path, xml_file_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ID = root.get("id")
        for document in root:
            document_df.loc[len(document_df)] = [ID, document.text.strip(" \"\n\'\t").lower()]
    return document_df            


def merge(data_path, label_df, document_df):
    id2age = dict(zip(label_df["ID"].tolist(), label_df["AGE"].tolist()))
    document_df.insert(2, 'AGE', None)
    print('csv making...')
    for i in tqdm(range(len(document_df))):
        document_df.loc[i, "AGE"] = id2age[document_df.loc[i, "ID"]]
    document_df.to_csv(os.path.join(data_path, "data_train.csv"), index=None)


if __name__ == '__main__':
    data_path = './dataset/Age/data_train/data_english'
    label_df = get_label_df(data_path)
    document_df = get_document_df(data_path)
    merge(data_path, label_df, document_df)
