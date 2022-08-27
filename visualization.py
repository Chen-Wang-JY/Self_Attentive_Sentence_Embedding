import torch
import argparse

from models import AttentionModel
from dataloader import YelpWordEmbedding


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default='./trained_models/model.pt')
    parser.add_argument("--heat_map_path", type=str, default='./html_heatmap/heatmap.html')
    parser.add_argument("--Yelp_data_path", type=str, default='./dataset/Yelp/my_train.csv')

    parser.add_argument("--sentence", type=str, default="I really enjoy Ashley and Ami salon she do a great job be friendly and professional I usually get my hair do when I go to MI because of quality of the highlight and the price the price be very affordable the hightlight fantastic thank Ashley i highly recommend you and i'll be back")
    parser.add_argument("--muti_sentence", type=bool, default=False)
    parser.add_argument("--path", type=str, default="./html_heatmap/sentence.txt")

    return parser.parse_args()


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def get_html_line(seq, attns):
    span = ""
    for ix, attn in zip(seq, attns):
        span += ' ' + highlight(
            ix,
            attn
        )
    return span


def mk_html(spans):
    header = "\t<head>\n\t\t<meta charset=\"UTF-8\">\n\t\t<title>heatmap</title>\n\t</head>\n"
    body = "\t<body>\n\t\t<h3>Visualization results</h3>\n\t\t{}\n\t</body>\n".format(spans)
    html = "<html>\n{}{}</html>".format(header, body)
    return html


if __name__ == '__main__':

    args = get_argparse()

    model = torch.load(args.model_path)
    word_embedding = YelpWordEmbedding(args.Yelp_data_path)

    if args.muti_sentence is True:
        sentences = []
        with open(args.path, 'r') as fp:
            for line in fp:
                sentences.append(line.strip().lower())
    else:
        sentences = [args.sentence.lower()]
    
    spans = ""
    for sentence in sentences:
        outs, A = model(torch.unsqueeze(word_embedding.get_vector(sentence), 0), retain_A=True)
        A = torch.squeeze(A.sum(dim=2))
        A = A / A.sum()
        spans += get_html_line(word_embedding.tokenize(sentence), A) + "<br><br>"
    html = mk_html(spans)

    with open(args.heat_map_path, 'w') as fp:
        fp.write(html)
    print("success")
    