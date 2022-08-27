import argparse
from models import AttentionModel, BiLSTM, CNNModel
from dataloader import WordEmbedding, YelpWordEmbedding
from dataloader import AgeDataset, YelpDataset

import torch
import torch.nn.functional as F
import logging

from utils import criterion, set_seed

def get_argparse():
    parser = argparse.ArgumentParser()

    # 任务
    parser.add_argument("--model_type", type=str, default="SelfAttention", help="BiLSTM or SelfAttention or CNN")
    parser.add_argument("--dataset", type=str, default="Yelp", help="Age or Yelp")
    parser.add_argument("--Age_train_path", type=str, default="./dataset/Age/data_train/data_english/data_train.csv")
    parser.add_argument("--Age_test_path", type=str, default="./dataset/Age/data_test/data_english/data_test.csv")
    parser.add_argument("--Yelp_train_path", type=str, default="./dataset/Yelp/my_train.csv")
    parser.add_argument("--Yelp_test_path", type=str, default="./dataset/Yelp/my_test.csv")


    # SelfAttention and BiLSTM model
    parser.add_argument("--LSTM_hidden", type=int, default=75)         # 300
    parser.add_argument("--MLP_hidden", type=int, default=100)         # 2000
    parser.add_argument("--attention_hidden", type=int, default=75)    # 350
    parser.add_argument("--aspects", type=int, default=10)              # 30
    parser.add_argument("--num_classes", type=int, default=5, help="Age should be 4 and Yelp should be 5")   

    # CNN model
    parser.add_argument("--out_channels", type=int, default=100)    # 一般与词向量维度一致
    parser.add_argument("--kernel_size", type=int, default=3)
    
    # 训练
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)           # 16
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0008)   # 0.0001
    parser.add_argument("--ratio", type=float, default=0.04)               # 1
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default="./trained_models/model.pt")

    # 其他
    parser.add_argument("--seed", type=int, default=20001202)
    parser.add_argument("--word_embedding_dimension", type=int, default=100)
    
    return parser.parse_args()


def train_epoch(args, epoch, train_loader, model, device):
    # optim = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.9)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)# , weight_decay=1e-5)

    total_loss = 0
    for index, (document, label) in enumerate(train_loader):
        document = document.to(device)
        label = label.to(device)

        optim.zero_grad()

        if args.model_type == 'SelfAttention':
            outs, A = model(document, retain_A=True)
            ce_loss, f_loss = criterion(model, label, outs, A, device, args)
            loss = ce_loss + args.ratio * f_loss
        elif args.model_type == 'BiLSTM' or args.model_type == 'CNN':
            outs = model(document)
            loss = F.cross_entropy(outs, label)

        loss.backward()
        optim.step()
        total_loss += loss.item()

    logging.info("EPOCH {}\tavgLoss: {:.4f}".format(epoch, total_loss / len(train_loader)))


def test(args, data_loader, model, device):
    count = 0
    with torch.no_grad():
        for index, (document, label) in enumerate(data_loader):
            document = document.to(device)
            label = label.to(device)

            outs = model(document)
            outs = torch.argmax(outs, 1)
            count += (outs == label).sum().item()
            
    return count / (len(data_loader) * args.batch_size)


def get_model(args, device):
    if args.model_type == 'BiLSTM':
        return BiLSTM(args).to(device)
    if args.model_type == 'SelfAttention':
        return AttentionModel(args).to(device)
    if args.model_type == 'CNN':
        return CNNModel(args).to(device)
    else:
        logging.info("model name illegal")
        exit()


def get_word_embedding(args):
    if args.dataset == 'Age':
        word_embedding = WordEmbedding(args.Age_train_path)
    elif args.dataset == 'Yelp':
        word_embedding = YelpWordEmbedding(args.Yelp_train_path)
    else:
        logging.info("dataset name illegal")
        exit()
    logging.info("词库建立完成")
    return word_embedding


def get_dataset(args, word_embedding):
    if args.dataset == 'Age':
        train_set = AgeDataset(args.Age_train_path, word_embedding)
        test_set = AgeDataset(args.Age_test_path, word_embedding)
    elif args.dataset == 'Yelp':
        train_set = YelpDataset(args.Yelp_train_path, word_embedding)
        test_set = YelpDataset(args.Yelp_test_path, word_embedding)
    logging.info("数据集加载成功")
    return train_set, test_set


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s')
    args = get_argparse()

    set_seed(args.seed)

    word_embedding = get_word_embedding(args)
    train_set, test_set = get_dataset(args, word_embedding)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=train_set.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_set.collate_fn
    )

    device = torch.device("cuda" if args.cuda else "cpu")

    model = get_model(args, device)

    logging.info(str(args) + '\n')
    logging.info("MODEL [{}]\n".format(args.model_type))
    logging.info("DATASET [{}]\n".format(args.dataset))
    logging.info("Train begins" + '\n')
    for e in range(args.epochs):
        train_epoch(args, e, train_loader, model, device)
        test_acc = test(args, test_loader, model, device)
        train_acc = test(args, train_loader, model, device)
        logging.info("EPOCH {}\ttrain accuracy {:.4f}\ttest acccuracy {:.4f}".format(e, train_acc, test_acc))
    
    logging.info("Train over")
    if args.save_model:
        model.to("cpu")
        torch.save(model, args.save_path)
        logging.info("model saves to " + args.save_path)

