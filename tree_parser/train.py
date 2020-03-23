import os
import random
import sys

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from tree_parser.model import GraphTransformerModel
from tree_parser.utils import create_data, batchify, test

_path = os.path.dirname(__file__)
_train_filename = os.path.join(_path, '../data/en_gum-ud-train.conllu')
_dev_filename = os.path.join(_path, '../data/en_gum-ud-dev.conllu')
_save_filename = os.path.join(_path, '../data/save')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')


def train(train_model, batches, optimizer, edges_criterion, adj_criterion, pos_criterion, tokens_criterion):
    total_loss = 0.
    for i, batch in enumerate(batches):
        inputs, edges_targets, adj_targets, pos_targets = batch[0], batch[1], batch[2], batch[3]
        optimizer.zero_grad()
        output_edges, output_adj, output_pos = train_model(inputs.cuda())
        loss_edges = edges_criterion(output_edges, edges_targets.cuda().float())
        loss_adj = adj_criterion(output_adj, adj_targets.cuda().float())
        loss_pos = pos_criterion(output_pos, pos_targets.cuda().float())
        loss = loss_edges + 2. * loss_adj + loss_pos
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    train_data = create_data(_train_filename, tokenizer, limit=None)
    train_batches = batchify(train_data, 10)

    dev_data = create_data(_dev_filename, tokenizer, limit=None)
    dev_batches = batchify(dev_data, 100)

    train_model = GraphTransformerModel(language_model)
    train_model.cuda()

    edges_criterion = nn.BCELoss()
    adj_criterion = nn.BCELoss()
    pos_criterion = nn.BCELoss()
    tokens_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=5e-5)

    best_epoch = 0
    best_value = -1
    for epoch in range(100):
        random.shuffle(train_batches)
        train_model.train()

        loss = train(train_model, train_batches, optimizer, edges_criterion, adj_criterion, pos_criterion,
                     tokens_criterion)

        print('Epoch:', epoch, 'Loss:', loss)

        train_model.eval()
        LAS_score, _, _ = test(train_model, dev_batches)

        if LAS_score > best_value:
            best_epoch = epoch
            best_value = LAS_score

        torch.save({
            'epoch': epoch,
            'model_state_dict': train_model.state_dict()},
            _save_filename + str(epoch))

        sys.stdout.flush()

    print('BEST EPOCH: ', best_epoch)
