import os

import torch
from transformers import BertModel, BertTokenizer

from tree_parser.model import GraphTransformerModel
from tree_parser.utils import create_data, batchify, test

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/tree_parser.model')
_test_filename = os.path.join(_path, '../data/en_gum-ud-dev.conllu')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')

if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    test_data = create_data(_test_filename, tokenizer, limit=None)
    test_batches = batchify(test_data, 100)

    model = GraphTransformerModel(language_model)
    checkpoint = torch.load(_save_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    LAS_score, _, _ = test(model, test_batches)
