import os
import pickle

if __name__ == '__main__':
    data_tag = 'tw_mm_s4'  # 'tw_mm_s1' || 'tw_mm_imagenet_s2' || 'tw_mm_daily_s2'
    data_dir = '../data/{}'.format(data_tag)
    for data_tag in ['train', 'valid', 'test']:
        print('\nComputing url map for %s' % data_tag)
        src_fn = os.path.join(data_dir, '{}_src.txt'.format(data_tag))
        trg_fn = os.path.join(data_dir, '{}_url_map.pt'.format(data_tag))
        url_map = {}
        with open(src_fn, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                img_fn = line.split('<sep>')[-1].split('/')[-1].strip()
                if img_fn not in url_map.keys():
                    url_map[img_fn] = idx
                else:
                    print('Error, there are duplicate img filenames: %s' % img_fn)
        with open(trg_fn, 'wb') as fw:
            pickle.dump(url_map, fw)
        print('Dump %d items of a dict into %s' % (len(url_map), trg_fn))
