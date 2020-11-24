import os
import pickle

if __name__ == '__main__':
    data_tag = 'tw_mm_imagenet_s2'  # 'tw_mm_s1' || 'tw_mm_imagenet_s2' || 'tw_mm_daily_s2'
    data_dir = '../data/{}'.format(data_tag)
    for data_tag in ['train', 'valid', 'test']:
        print('\nComputing src_str map for %s' % data_tag)
        src_fn = os.path.join(data_dir, '{}_src.txt'.format(data_tag))
        trg_fn = os.path.join(data_dir, '{}_src_str_map.pt'.format(data_tag))
        src_str_map = {}
        with open(src_fn, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                src_str = line.split('<sep>')[0].strip()
                if src_str not in src_str_map.keys():
                    src_str_map[src_str] = idx
                else:
                    print('Error, there are duplicate src str: %s' % src_str)
        with open(trg_fn, 'wb') as fw:
            pickle.dump(src_str_map, fw)
        print('Dump %d items of a dict into %s' % (len(src_str_map), trg_fn))
