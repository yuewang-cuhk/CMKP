import re
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import time


def get_text_processor():
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "elongated", "repeated", 'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    return text_processor


def write_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def tw_tokenize(src_fn, trg_fn, ocr_dict, valid_token_set):
    text_processor = get_text_processor()
    t0 = time.time()
    def process_sentence(text, text_processor=text_processor):
        # return a list of tokens
        tokens = text_processor.pre_process_doc(text)
        filter_set = {'<repeated>', '<elongated>'}
        return [t for t in tokens if t not in filter_set]

    valid_idx = 0
    fw = open(trg_fn, 'w', encoding='utf-8')
    with open(src_fn, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            img_fn = line.split('/')[-1].strip()
            if img_fn not in ocr_dict:
                fw.write('\n')
                continue
            ocr_text = ocr_dict[img_fn]
            tokens = process_sentence(ocr_text)
            tokens = list(filter(lambda x: re.match('^[a-zA-Z]+$', x), tokens))
            tokens = list(filter(lambda x: x in valid_token_set and len(x) > 1, tokens))

            long_num = sum([len(t) > 3 for t in tokens])
            short_num = sum([len(t) < 3 for t in tokens])
            if len(tokens) == 0 or short_num / len(tokens) >= 0.75 or long_num < 3:
                fw.write('\n')
            else:
                fw.write(' '.join(tokens) + '\n')
                valid_idx += 1
    fw.close()
    idx += 1
    print('Writing %d tweets into %s, takes %.2f seconds' % (idx, trg_fn, time.time() - t0))
    print('Valid OCR rate is %d/%d = %.2f' % (valid_idx, idx, valid_idx / idx))


if __name__ == '__main__':
    t0 = time.time()
    ocr_fn = 'CMKP_ocr.txt'
    src_fn = '../data/tw_mm_s3/{}_src.txt'
    trg_fn = '../data/tw_mm_s3/{}_ocr_final.txt'

    all_tokens = []
    for tag in ['train', 'valid', 'test']:
        src_lines = open(src_fn.format(tag), 'r').readlines()
        for line in src_lines:
            tags = line.strip().split('<sep>')[1].strip().split(';')
            tag_tokens = []
            for tag in tags:
                tag_tokens.extend(tag.split())
            post_tokens = line.strip().split('<sep>')[0].split()
            all_tokens += tag_tokens + post_tokens

    valid_token_set = set(all_tokens)
    print('The size of valid token set: %d, takes %.2f seconds' % (len(valid_token_set), time.time() - t0))

    ocr_dict = dict()
    with open(ocr_fn, 'r') as f:
        for idx, line in enumerate(f):
            sep_id = line.index(':')
            k = line[:sep_id].strip()
            v = line[sep_id + 1:].strip().replace('\n', '').replace('\r', '')
            if len(v) != 0:
                ocr_dict[k] = v
    print('There are %d/%d in %s has OCR, takes %.2f seconds' % (len(ocr_dict), idx + 1, ocr_fn, time.time() - t0))
    # There are 49621/146656 in CMKP_ocr.txt has OCR, takes 3.27 seconds

    for tag in ['train', 'valid', 'test']:
        cur_src_fn = src_fn.format(tag)
        cur_trg_fn = trg_fn.format(tag)
        tw_tokenize(cur_src_fn, cur_trg_fn, ocr_dict, valid_token_set)
