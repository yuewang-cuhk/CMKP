import os
import time


def run_mixture_multi_head_tiny(is_debug=False):
    t0 = time.time()
    total_cnt = 0
    fail_cnt = 0
    fail_cmd = []
    model_zoo = [
        'mixture_img_text_multi_head_att_h4_d128',
    ]

    res_fn = 'results/mixture_multi_head_tiny_pretrain.csv'
    for seed in [23]:
        for cur_model in model_zoo:
            t1 = time.time()
            cmd_str = 'python ../unified_model/train.py -cur_model %s -data_tag tw_mm_s1_tiny -seed %d  -res_fn %s ' % \
                      (cur_model, seed, res_fn)
            if is_debug:
                cmd_str += ' -debug 1'
            try:
                print('\n\n\n============================Start Running (%d)==========================' % total_cnt)
                print('%s\n' % cmd_str)
                os.system(cmd_str)
                total_cnt += 1
            except:
                print('Error in %s' % cmd_str)
                fail_cnt += 1
                fail_cmd.append(cmd_str)
                continue
            print('\n\nFinish run [%s] and takes %.2f' % (cmd_str, time.time() - t1))
    print('\n\nIn total finish %d/%d commands and takes %.2f' % (total_cnt - fail_cnt, total_cnt, time.time() - t0))
    if fail_cnt > 0:
        print('The fail command list:\n', fail_cmd)


def run_mixture_multi_head_ocr_combine_direct_tiny(is_debug=False):
    t0 = time.time()
    total_cnt = 0
    fail_cnt = 0
    fail_cmd = []
    model_path = 'models/mixture_img_text_multi_head_att_h4_d128-copy_vgg_glove_s23_1124-1210_3.799M-tw_mm_s1_tiny/e3_TL31.09_VL5.98_0h-00m.ckpt'
    cur_model = 'mixture_img_text_multi_head_att_h4_d128_combine_direct'
    res_fn = 'results/mixture_multi_head_ocr_combine_direct_tiny.csv'
    for seed in [24]:
        t1 = time.time()
        cmd_str = 'python ../unified_model/train.py -cur_model %s  -model_path %s -data_tag tw_mm_s1_tiny -seed %d -fix_classifier 1 -res_fn %s ' % \
                  (cur_model, model_path, seed, res_fn)
        if is_debug:
            cmd_str += ' -debug 1'
        try:
            print('\n\n\n============================Start Running (%d)==========================' % total_cnt)
            print('%s\n' % cmd_str)
            os.system(cmd_str)
            total_cnt += 1
        except:
            print('Error in %s' % cmd_str)
            fail_cnt += 1
            fail_cmd.append(cmd_str)
            continue
        print('\n\nFinish run [%s] and takes %.2f' % (cmd_str, time.time() - t1))
    print('\n\nIn total finish %d/%d commands and takes %.2f' % (total_cnt - fail_cnt, total_cnt, time.time() - t0))
    if fail_cnt > 0:
        print('The fail command list:\n', fail_cmd)


def run_mixture_multi_head_ocr(is_debug=False):
    t0 = time.time()
    total_cnt = 0
    fail_cnt = 0
    fail_cmd = []
    model_zoo = [
        'mixture_img_text_multi_head_att_h4_d128',
    ]

    res_fn = 'results/mixture_multi_head_ocr_pretrain.csv'
    for seed in [23]:
        for cur_model in model_zoo:
            t1 = time.time()
            cmd_str = 'python ../unified_model/train.py -cur_model %s -data_tag tw_mm_s1_ocr -seed %d  -res_fn %s ' % \
                      (cur_model, seed, res_fn)
            if is_debug:
                cmd_str += ' -debug 1'
            try:
                print('\n\n\n============================Start Running (%d)==========================' % total_cnt)
                print('%s\n' % cmd_str)
                os.system(cmd_str)
                total_cnt += 1
            except:
                print('Error in %s' % cmd_str)
                fail_cnt += 1
                fail_cmd.append(cmd_str)
                continue
            print('\n\nFinish run [%s] and takes %.2f' % (cmd_str, time.time() - t1))
    print('\n\nIn total finish %d/%d commands and takes %.2f' % (total_cnt - fail_cnt, total_cnt, time.time() - t0))
    if fail_cnt > 0:
        print('The fail command list:\n', fail_cmd)


def run_mixture_multi_head_ocr_combine_direct(is_debug=False):
    t0 = time.time()
    total_cnt = 0
    fail_cnt = 0
    fail_cmd = []
    model_path = 'sample_model/e10_joint_pretrain.ckpt'
    cur_model = 'mixture_img_text_multi_head_att_h4_d128_combine_direct'
    res_fn = 'results/mixture_multi_head_ocr_combine_direct_seed.csv'
    for seed in [24]:
        t1 = time.time()
        cmd_str = 'python ../unified_model/train.py -cur_model %s  -model_path %s -data_tag tw_mm_s1_ocr -seed %d -fix_classifier 1 -res_fn %s ' % \
                  (cur_model, model_path, seed, res_fn)
        if is_debug:
            cmd_str += ' -debug 1'
        try:
            print('\n\n\n============================Start Running (%d)==========================' % total_cnt)
            print('%s\n' % cmd_str)
            os.system(cmd_str)
            total_cnt += 1
        except:
            print('Error in %s' % cmd_str)
            fail_cnt += 1
            fail_cmd.append(cmd_str)
            continue
        print('\n\nFinish run [%s] and takes %.2f' % (cmd_str, time.time() - t1))
    print('\n\nIn total finish %d/%d commands and takes %.2f' % (total_cnt - fail_cnt, total_cnt, time.time() - t0))
    if fail_cnt > 0:
        print('The fail command list:\n', fail_cmd)


if __name__ == '__main__':
    run_mixture_multi_head_ocr_combine_direct()
    # run_mixture_multi_head_ocr()
    # run_mixture_multi_head_ocr_tiny()
    # run_mixture_multi_head_ocr_combine_direct_tiny()
