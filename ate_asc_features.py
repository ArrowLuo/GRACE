from __future__ import absolute_import, division, print_function

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_labels(label_tp_list):
    at_labels = []
    as_labels = []
    for l in label_tp_list:
        if l[0] not in at_labels:
            at_labels.append(l[0])
        if l[1] not in as_labels:
            as_labels.append(l[1])
    if "O" in at_labels:
        at_labels.remove("O")
    if "O" in as_labels:
        as_labels.remove("O")
    at_labels.insert(0, "O")
    as_labels.insert(0, "O")
    at_labels = [l.replace("_", "-") for l in at_labels]
    return at_labels, as_labels

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label_tp=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_tp = label_tp

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, at_label_id, as_label_id,
                 label_mask, label_mask_X):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.at_label_id = at_label_id
        self.as_label_id = as_label_id
        self.label_mask = label_mask
        self.label_mask_X = label_mask_X

class ATEASCProcessor():
    def __init__(self, file_path, set_type):
        corpus_tp, label_tp_list = self._readfile(file_path)
        examples = self._create_examples(corpus_tp, set_type)
        self.examples = examples
        self.label_tp_list = label_tp_list

    def _readfile(self, filename):
        f = open(filename)
        data = []
        labels = []

        sentence = []
        label = []
        for line in f:
            line = line.strip()
            line = line.replace("\t", " ")
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    labels += label
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append((splits[-3], splits[-2]))

        if len(sentence) > 0:
            data.append((sentence, label))
            labels += label
            sentence = []
            label = []
        return data, labels

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label_tp) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            at_label = [l[0].replace("_", "-") for l in label_tp]
            as_label = [l[1].replace("_", "-") for l in label_tp]
            label_tp = (at_label, as_label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label_tp=label_tp))
        return examples

def convert_examples_to_features(examples, label_tp_list, max_seq_length, tokenizer, verbose_logging=False):
    at_label_list, as_label_list = label_tp_list
    at_label_map = {label: i for i, label in enumerate(at_label_list)}
    as_label_map = {label: i for i, label in enumerate(as_label_list)}

    # Note: below contains hard code on "B-AP", "I-AP", and "O"
    assert all(c_str in at_label_map.keys() for c_str in ["B-AP", "I-AP", "O"])

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        label_tp_list = example.label_tp
        tokens = []
        at_labels = []
        as_labels = []

        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_tp_list[0][i]
            label_2 = label_tp_list[1][i]
            for m in range(len(token)):
                if m == 0:
                    at_labels.append(label_1)
                    as_labels.append(label_2)
                else:
                    at_labels.append("X")
                    as_labels.append(label_2)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            at_labels = at_labels[0:(max_seq_length - 2)]
            as_labels = as_labels[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        at_label_ids = []
        as_label_ids = []
        label_mask = []
        label_mask_X = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        at_label_ids.append(-1)
        as_label_ids.append(-1)
        label_mask.append(-1)
        label_mask_X.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if at_labels[i] == "X":
                at_label_ids.append(-1)
                label_mask.append(-1)
                label_mask_X.append(1)
            else:
                at_label_ids.append(at_label_map[at_labels[i]])
                label_mask.append(len(ntokens)-1)
                label_mask_X.append(0)
            if as_labels[i] == "X":
                as_label_ids.append(-1)
            else:
                as_label_ids.append(as_label_map[as_labels[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        at_label_ids.append(-1)
        as_label_ids.append(-1)
        label_mask.append(-1)
        label_mask_X.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            at_label_ids.append(-1)
            as_label_ids.append(-1)
            label_mask.append(-1)
            label_mask_X.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(at_label_ids) == max_seq_length
        assert len(as_label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(label_mask_X) == max_seq_length

        if verbose_logging and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("at_label: %s (id = %s)" % (example.label_tp, " ".join([str(x) for x in at_label_ids])))
            logger.info("as_label: %s (id = %s)" % (example.label_tp, " ".join([str(x) for x in as_label_ids])))
            logger.info("label_mask: %s" % (" ".join([str(x) for x in label_mask])))
            logger.info("label_mask_X: %s" % (" ".join([str(x) for x in label_mask_X])))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                          at_label_id=at_label_ids, as_label_id=as_label_ids,
                          label_mask=label_mask, label_mask_X=label_mask_X))

    return features