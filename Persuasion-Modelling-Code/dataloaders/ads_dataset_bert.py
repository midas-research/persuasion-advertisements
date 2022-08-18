from re import I
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import nltk
import numpy as np
import json
from random import randrange

from dataloaders.utils import load_vocab
from dataloaders.utils import load_raw_annots
from dataloaders.utils import load_action_reason_annots
from dataloaders.utils import load_densecap_annots
from dataloaders.utils import load_symbol_cluster
from dataloaders.utils import tokenize


class AdsDataset(Dataset):
    def __init__(self, config, split='train'):

        print("Initializing dataset:")

        # Image features.
        self.image_features = np.load(config['image_feature_path'], allow_pickle=True, encoding='bytes').item()
        self.region_features = np.load(config['region_feature_path'], allow_pickle=True, encoding='bytes').item()
        print("Image features loaded, img_len={}, roi_len={}.".format(len(self.image_features),
                                                                      len(self.region_features)))

        # print("Img features", self.image_features)

        # Action-reason annotations
        self.stmt_annots = np.load("output/stmt_features_bert_train.npy", allow_pickle=True, encoding='bytes').item()
        print("Annotations are loaded, len={}.".format(len(self.stmt_annots)))

        # Densecap annotations.
        self.dense_annots = load_densecap_annots(config['densecap_annot_path'], config['max_densecaps_per_image'])
        self.dense_vocab = load_vocab(config['densecap_vocab_path'])
        print("Densecap annotations are loaded, len={}.".format(len(self.dense_annots)))
        print("Loaded vocab from {}, vocab size={}.".format(config['densecap_vocab_path'], len(self.dense_vocab)))

        # Symbol annotations
        self.symbol_annots = load_raw_annots(config['symbol_annot_path'])
        self.word_to_id, self.id_to_symbol = load_symbol_cluster(config['symbol_cluster_path'])
        print("Symbol annotations are loaded, len={}.".format(len(self.symbol_annots)))

        self.split = split

        # Initialize feed_dict.
        self.annot_dicts = []

        total_images = total_statements = 0

        # Split training data for validation purpose.
        self.stmt_annots = self.stmt_annots.items()
        if split == 'val':
            self.stmt_annots = list(self.stmt_annots)[:config['number_of_val_examples']]
        elif split == 'train':
            self.stmt_annots = list(self.stmt_annots)[config['number_of_val_examples']:]

        print("Processing {} {} records".format(len(self.stmt_annots), split))

        for index, (image_id, annot) in enumerate(self.stmt_annots):

            # Pad action-reason.
            (number_of_statements, statement_strings, statement_lengths) = self.encode_bert(annot[0], config['max_stmts_per_image'], 768)

            # Pad densecap.
            if not config['use_single_densecap']:
                (number_of_densecaps, densecap_strings, densecap_lengths) = self.encode_and_pad_sentences(
                    self.dense_vocab, self.dense_annots[image_id], config['max_densecaps_per_image'],
                    config['max_densecap_len'])

            else:  # Concatenate all densecaps to form a single sentence.
                dense_string_concat = ' '.join(self.dense_annots[image_id])
                (number_of_densecaps, densecap_strings, densecap_lengths) = self.encode_and_pad_sentences(
                    self.dense_vocab, [dense_string_concat], 1,
                    config['max_densecap_len'] * config['max_densecaps_per_image'])

            # Pad symbols.
            symbols = self.symbol_annots.get(image_id, [])
            number_of_symbols = len(symbols)
            symbols += [0] * config['max_symbols_per_image']
            symbols = symbols[:config['max_symbols_per_image']]

            feed_dict = {}
            feed_dict['image_id'] = image_id
            feed_dict['img_features'] = self.image_features[image_id]
            feed_dict['roi_features'] = self.region_features[image_id]
            feed_dict['number_of_statements'] = np.array(number_of_statements, dtype=np.int32)
            feed_dict['statement_strings'] = statement_strings
            feed_dict['statement_lengths'] = statement_lengths
            feed_dict['number_of_densecaps'] = np.array(number_of_densecaps, dtype=np.int32)
            feed_dict['densecap_strings'] = densecap_strings
            feed_dict['densecap_lengths'] = densecap_lengths
            feed_dict['number_of_symbols'] = np.array(number_of_symbols, dtype=np.int32)
            feed_dict['symbols'] = np.array(symbols)

            # Removing inf values
            feed_dict['img_features'][np.isinf(feed_dict['img_features'])] = 0
            feed_dict['img_features'][np.isnan(feed_dict['img_features'])] = 0

            if split != 'x':
                # Pad strings for evaluation purpose.
                (number_of_eval_statements, eval_statement_strings, 
                eval_statement_lengths) = self.encode_bert(annot[1], config['number_of_val_stmts_per_image'], 768)

                assert number_of_eval_statements == config['number_of_val_stmts_per_image']
                feed_dict['eval_statement_strings'] = eval_statement_strings
                feed_dict['eval_statement_lengths'] = eval_statement_lengths

            self.annot_dicts.append(feed_dict)
            total_images += 1
            total_statements += number_of_statements

            if index % 1000 == 0:
                print("Load on {}/{}".format(index, len(self.stmt_annots)))

        if split != 'test':
            self.annot_dicts = list(filter(lambda x: x['number_of_statements'] > 0, self.annot_dicts))

    def encode_and_pad_sentences(self, vocab, sentences, max_sents_per_image, max_sent_len):
        """Encodes and pads sentences.

        Args:
            vocab: a dict mapping from word to id.
            sentences: a list of python string.
            max_sents_per_image: maximum number of sentences.
            max_sent_len: maximum length of sentence.

        Returns:
            num_sents: a integer denoting the number of sentences.
            sent_mat: a [max_sents_per_image, max_sent_len] numpy array pad with zero.
            sent_len: a [max_sents_per_image] numpy array indicating the length of each
            sentence in the matrix.
        """
        encode_fn = lambda x: [vocab.get(w, 0) for w in tokenize(x)]

        sentences = [encode_fn(s) for s in sentences]
        sent_mat = np.zeros((max_sents_per_image, max_sent_len), np.int32)
        sent_len = np.zeros((max_sents_per_image,), np.int32)

        for index, sent in enumerate(sentences[:max_sents_per_image]):
            sent_len[index] = min(max_sent_len, len(sent))
            sent_mat[index][:sent_len[index]] = sent[:sent_len[index]]

        return len(sentences), sent_mat, sent_len

    def encode_bert(self,sentences, max_sents_per_image, enc_dim):
        """Encodes and pads sentences.

        Args:
            sentences: a list of [768] size encoding of the sentences
            max_sents_per_image: maximum number of sentences.
            enc_dim: encoding dimension.

        Returns:
            num_sents: a integer denoting the number of sentences.
            sent_mat: a [max_sents_per_image, enc_dim] numpy array 
            sent_dim : a [max_sents_per_image] numpy array
        """
        sent_mat = np.zeros((max_sents_per_image, enc_dim), np.float32)
        sent_len = np.zeros((max_sents_per_image,), np.int32)

        for index, sent in enumerate(sentences[:max_sents_per_image]):
            sent_len[index] = enc_dim
            sent_mat[index] = sent

        return len(sentences), sent_mat, sent_len


    def __len__(self):
        return len(self.annot_dicts)

    def __getitem__(self, idx):
        annot = self.annot_dicts[idx]
        data = {}

        if annot['number_of_statements'] > 0:
            index = randrange(0, annot['number_of_statements'])
        else:
            index = 0

        data['statement_strings'] = annot['statement_strings'][index]
        data['statement_lengths'] = annot['statement_lengths'][index]

        index = randrange(0, annot['number_of_densecaps'])
        data['densecap_strings'] = annot['densecap_strings'][index]
        data['densecap_lengths'] = annot['densecap_lengths'][index]

        data['img_features'] = torch.from_numpy(annot['img_features'])
        data['roi_features'] = torch.from_numpy(annot['roi_features'])

        data['statement_strings'] = torch.from_numpy(data['statement_strings'])
        data['densecap_strings'] = torch.from_numpy(data['densecap_strings'])
        data['symbols'] = torch.from_numpy(annot['symbols'])

        data['image_id'] = annot['image_id']
        data['number_of_statements'] = annot['number_of_statements']
        data['number_of_densecaps'] = annot['number_of_densecaps']
        data['number_of_symbols'] = annot['number_of_symbols']

        if self.split != 'x':
            data['eval_statement_strings'] = torch.from_numpy(annot['eval_statement_strings'])
            data['eval_statement_lengths'] = torch.from_numpy(annot['eval_statement_lengths'])

        return data


if __name__ == '__main__':
    with open('configs/advise_densecap_data.json') as fp:
        config = json.load(fp)
    dataset = AdsDataset(config)

    print(dataset[0])
