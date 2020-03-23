import os

import torch
from networkx import DiGraph
from networkx.drawing.nx_agraph import write_dot
from transformers import BertModel, BertTokenizer

from tree_parser.model import GraphTransformerModel
from tree_parser.utils import create_data_from_sentences, batchify_sentences, pos_tags, dep_edges

_path = os.path.dirname(__file__)

_save_filename = os.path.join(_path, '../data/tree_parser.model')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')

_text = """
In nuclear physics, the island of stability is a predicted set of isotopes of superheavy elements 
that may have considerably longer half-lives than known isotopes of these elements. 
"""

def create_graph(edges, adj, pos, tokens):
    tokens = tokenizer.convert_ids_to_tokens(tokens[1:])

    words = []
    joined_word = ''
    for token in tokens[::-1]:
        words.append(token + joined_word)
        if token[:2] == '##':
            joined_word = token[2:] + joined_word
        else:
            joined_word = ''
    words = words[::-1]

    pos_indices = [torch.argmax(vector) for vector in pos[1:]]
    edges_indices = [torch.argmax(vector) for vector in edges[1:]]
    adj_indices = [int(torch.argmax(vector)) - 1 for vector in adj][1:]

    pos_labels = [pos_tags[index] for index in pos_indices]
    edge_labels = [dep_edges[index] for index in edges_indices]


    g = DiGraph()
    for index, item in enumerate(zip(pos_labels, edge_labels, words, adj_indices)):
        pos = item[0]
        edge = item[1]
        token = item[2]
        link_from = item[3]

        if pos == 'JOIN':
            continue

        g.add_node(str(index) + '_' + token, **{'pos': pos, 'token': token})
        if link_from != -1:
            g.add_edge(str(index) + '_' + token,
                       str(link_from) + '_' + words[link_from],
                       **{'label': edge})

    return g


def predict(model, batch):
    output_edges, output_adj, output_pos = model(batch.cuda())

    graph_list = []
    for edges, adj, pos, tokens in zip(output_edges, output_adj, output_pos, batch):
        graph_list.append(create_graph(edges, adj, pos, tokens))

    return graph_list


if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    model = GraphTransformerModel(language_model)
    checkpoint = torch.load(_save_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    data = create_data_from_sentences([_text], tokenizer)
    batches = batchify_sentences(data, 10)

    for batch in batches:
        g_list = predict(model, batch)

    write_dot(g_list[0], 'test.dot')