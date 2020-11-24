import torch
import numpy as np
import argparse
import pickle
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import BertModel, BertTokenizer

MAX_LEN = 100


def main(args):
    # Build models
    bert = BertModel.from_pretrained('bert-base-uncased',
                                     output_hidden_states=False,
                                     output_attentions=False).eval()
    bert = bert.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Finish restore the model!')

    for type_tag in ['train', 'valid', 'test']:  # 'train', 'valid', 'test'
        t0 = time.time()
        src_fn = args.src_fn.format(type_tag)
        pred_fn = args.pred_fn.format(type_tag)

        src_lines = [line.split('<sep>')[0].strip() for line in open(src_fn, 'r', encoding='utf-8')]

        # Load the trained model parameters\
        file_len = len(src_lines)
        print('\nBegin inferring for %d instances with batch_size: %d' % (file_len, args.batch_size))
        bert_feats = []
        ceil_file_len = ((file_len // args.batch_size) + 1) * args.batch_size if file_len % args.batch_size != 0 \
            else file_len
        print('Ceil file len: %d, batch num: %d' % (ceil_file_len, ceil_file_len //  args.batch_size))
        for b_id in range(0, ceil_file_len, args.batch_size):
            inputs = src_lines[b_id:b_id + args.batch_size]

            bert_src = [tokenizer.encode(input)[:MAX_LEN] for input in inputs]

            for b_ids in bert_src:
                if len(b_ids) < MAX_LEN:
                    b_ids.extend([0] * (MAX_LEN - len(b_ids)))

            bert_src = torch.LongTensor(bert_src).to(device)

            outputs = bert(bert_src)[1].detach().cpu().numpy()

            bert_feats.append(outputs)
            if (b_id // args.batch_size) % 50 == 0:
                print('Processing %d/%d instances, takes %.2f seconds' % (b_id, ceil_file_len, time.time() - t0))

        print('Finish inferring BERT features for %s and take %.2f seconds' % (src_fn, time.time() - t0))

        bert_feat = np.concatenate(bert_feats, axis=0)
        print('The size of the final BERT features: %s' % str(bert_feat.shape))
        with open(pred_fn, 'wb') as f:
            pickle.dump(bert_feat, f, protocol=4)

        print('Write %s of BERT features into %s' % (str(bert_feat.shape), pred_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_fn', type=str, default='../data/tw_mm_s1/{}_src.txt', help='')
    parser.add_argument('--pred_fn', type=str, default='../data/tw_mm_s1/{}_bert.pt', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    args = parser.parse_args()
    main(args)
