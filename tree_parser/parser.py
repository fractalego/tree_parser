import torch
from transformers import BertModel, BertTokenizer

from tree_parser.model import GraphTransformerModel
from tree_parser.utils import create_data_from_sentences, batchify_sentences, create_graph


class DependencyParser:
    _MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')
    _model_class, _tokenizer_class, _pretrained_weights = _MODEL
    _tokenizer = _tokenizer_class.from_pretrained(_pretrained_weights)
    _language_model = _model_class.from_pretrained(_pretrained_weights)

    def __init__(self, filename, num_batches=100):
        self._num_batches = num_batches
        self._model = GraphTransformerModel(self._language_model)
        checkpoint = torch.load(filename)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.cuda()

    def parse(self, text):
        return self.parse_list([text])[0]

    def parse_list(self, text_list):
        data = create_data_from_sentences(text_list, self._tokenizer)
        batches = batchify_sentences(data, 10)

        g_list = []
        for batch in batches:
            g_list += self._predict(batch)

        return g_list

    def _predict(self, batch):
        output_edges, output_adj, output_pos = self._model(batch.cuda())

        graph_list = []
        for edges, adj, pos, tokens in zip(output_edges, output_adj, output_pos, batch):
            graph_list.append(create_graph(edges, adj, pos, tokens, self._tokenizer))

        return graph_list