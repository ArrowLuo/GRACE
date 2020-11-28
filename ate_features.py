from __future__ import absolute_import, division, print_function

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_labels(label_list):
    labels = []
    for l in label_list:
        if l not in labels:
            labels.append(l)
    if "O" in labels:
        labels.remove("O")
    labels.insert(0, "O")
    labels = [l.replace("_", "-") for l in labels]
    return labels

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class ATEProcessor():
    def __init__(self, file_path, set_type):
        corpus, label_list = self._readfile(file_path)
        examples = self._create_examples(corpus, set_type)
        self.examples = examples
        self.label_list = label_list

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
            label.append(splits[-3])

        if len(sentence) > 0:
            data.append((sentence, label))
            labels += label
            sentence = []
            label = []
        return data, labels

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = [l.replace("_", "-") for l in label]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, verbose_logging=False):
    label_map = {label: i for i, label in enumerate(label_list)}

    # Note: below contains hard code on "B-AP", "I-AP", and "O"
    assert all(c_str in label_map.keys() for c_str in ["B-AP", "I-AP", "O"])

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []

        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(-1)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if labels[i] == "X":
                label_ids.append(-1)
            else:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(-1)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if verbose_logging and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_ids))
    return features