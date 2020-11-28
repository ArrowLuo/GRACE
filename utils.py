import logging
from collections import Counter

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def get_chunk_type(tag_name):
    tag_name_splits = tag_name.split('-')
    return tag_name_splits[-1], tag_name_splits[0]

def get_aspect_chunks(seq, default="O"):
    """
    Args:
        seq: [B-AP, O, B-AP, B-AP, I-AP, O, B-AP, I-AP, I-AP, O, B-AP] sequence of labels
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [B-AP, O, B-AP, B-AP, I-AP, O, B-AP, I-AP, I-AP, O, B-AP]
        result = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    """
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_type, tok_chunk_alpha = get_chunk_type(tok)
            if chunk_type is None and tok_chunk_alpha == "B":
                chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
                if tok_chunk_alpha == "B":
                    chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type == chunk_type:
                if tok_chunk_alpha == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def get_polaity_chunks(seq, aspect_lab_chunks, default="O", must_predict=False):
    """
    Args:
        seq: [POSITIVE, O, POSITIVE, POSITIVE, NEUTRAL, O, POSITIVE, NEUTRAL, NEUTRAL, O, POSITIVE] sequence of labels
        aspect_lab_chunks: [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = ["POSITIVE", "O", "POSITIVE", "POSITIVE", "NEUTRAL", "O", "POSITIVE", "NEUTRAL", "NEUTRAL", "O", "POSITIVE"]
        aspect_lab_chunks = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
        result = [('POSITIVE', 0, 1), ('POSITIVE', 2, 3), ('POSITIVE', 3, 5), ('NEUTRAL', 6, 9), ('POSITIVE', 10, 11)]
    """
    chunks = []
    for i, chunk in enumerate(aspect_lab_chunks):
        segs = seq[chunk[1]:chunk[2]]
        ins_counter = Counter(segs)
        if len(ins_counter) > 1:
            chunk_type = ins_counter.most_common(1)[0][0]
            if default != chunk_type:
                chunk_type = ins_counter.most_common(2)[1][0]
        else:
            chunk_type = ins_counter.most_common(1)[0][0]
        if must_predict or default != chunk_type:
            chunk = (chunk_type, chunk[1], chunk[2])
            chunks.append(chunk)
    return chunks