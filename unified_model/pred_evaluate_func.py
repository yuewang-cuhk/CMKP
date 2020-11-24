import argparse
from collections import defaultdict
import time
from metric_utils import *


def pred_evaluate(pred_fn, src_fn='data/tw_mm_s1/test_src.txt', res_fn='auto_results.csv',  # line_idx=[],
                  ocr_only=False, ocr_fn=None, verbose=False):
    print('\n================================Begin evaluate=================================')
    cmd_str = 'python ../pred_evaluate_func.py -pred %s -src %s -res_fn %s' % (pred_fn, src_fn, res_fn)
    print('Command: %s\n' % cmd_str)
    score_dict = defaultdict(list)
    topk_dict = {'present': [1, 3, 5], 'absent': [1, 3, 5], 'all': [1, 3, 5]}  # 'absent': [5, 10, 50, 'M']
    num_src = 0
    num_unique_predictions = 0
    num_present_filtered_predictions = 0
    num_present_unique_targets = 0
    num_absent_filtered_predictions = 0
    num_absent_unique_targets = 0
    max_unique_targets = 0

    t0 = time.time()
    src_lines = open(src_fn, 'r').readlines()
    pred_lines = open(pred_fn, 'r').readlines()
    if 'debug' in pred_fn:
        src_lines = src_lines[:100]
    assert len(src_lines) == len(pred_lines), 'the length of src do not equal to prediction'

    if ocr_only:
        ocr_lines = open(ocr_fn, 'r').readlines()
        assert len(ocr_lines) == len(src_lines)

    for data_idx, (src_l, pred_l) in enumerate(zip(src_lines, pred_lines)):
        if ocr_only and len(ocr_lines[data_idx].strip()) == 0:
            continue

        # if len(line_idx) != 0:
        #     if data_idx not in line_idx:
        #         continue
        num_src += 1
        # convert the str to token list
        pred_str_list = pred_l.strip().split(';')
        pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred_str_list]

        trg_str_list = src_l.strip().split('<sep>')[1].strip().split(';')
        trg_token_2dlist = [trg_str.strip().split(' ') for trg_str in trg_str_list]

        src_token_list = src_l.strip().split('<sep>')[0].strip().split()

        num_predictions = len(pred_str_list)

        # perform stemming
        stemmed_src_token_list = stem_word_list(src_token_list)
        stemmed_trg_token_2dlist = stem_str_list(trg_token_2dlist)
        stemmed_pred_token_2dlist = stem_str_list(pred_token_2dlist)

        # Filter out duplicate, invalid, and extra one word predictions
        filtered_stemmed_pred_token_2dlist, num_duplicated_predictions = filter_prediction(False,
                                                                                           True,
                                                                                           stemmed_pred_token_2dlist)
        num_unique_predictions += (num_predictions - num_duplicated_predictions)

        # Remove duplicated targets
        unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)
        max_unique_targets += (num_predictions - num_duplicated_trg)

        current_unique_targets = len(unique_stemmed_trg_token_2dlist)
        if current_unique_targets > max_unique_targets:
            max_unique_targets = current_unique_targets

        # separate present and absent keyphrases
        present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist = separate_present_absent_by_source(
            stemmed_src_token_list, filtered_stemmed_pred_token_2dlist)

        present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist = separate_present_absent_by_source(
            stemmed_src_token_list, unique_stemmed_trg_token_2dlist)

        num_present_filtered_predictions += len(present_filtered_stemmed_pred_token_2dlist)
        num_present_unique_targets += len(present_unique_stemmed_trg_token_2dlist)
        num_absent_filtered_predictions += len(absent_filtered_stemmed_pred_token_2dlist)
        num_absent_unique_targets += len(absent_unique_stemmed_trg_token_2dlist)

        # compute all the metrics and update the score_dict
        score_dict = update_score_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                       topk_dict['all'], score_dict, 'all')
        # compute all the metrics and update the score_dict for present keyphrase
        if present_unique_stemmed_trg_token_2dlist:
            score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist,
                                           present_filtered_stemmed_pred_token_2dlist,
                                           topk_dict['present'], score_dict, 'present')
        # compute all the metrics and update the score_dict for present keyphrase
        if absent_unique_stemmed_trg_token_2dlist:
            score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist,
                                           absent_filtered_stemmed_pred_token_2dlist,
                                           topk_dict['absent'], score_dict, 'absent')

        if (data_idx + 1) % 1000 == 0:
            print("Processing %d lines and takes %.2f" % (data_idx + 1, time.time() - t0))

    num_unique_targets = num_present_unique_targets + num_absent_unique_targets
    num_filtered_predictions = num_present_filtered_predictions + num_absent_filtered_predictions

    print('\nFinish evaluating for %d/%d instances' % (num_src, data_idx + 1))
    result_txt_str = ""

    # report global statistics
    result_txt_str += ('Total #samples: %d\n' % num_src)
    result_txt_str += ('Max. unique targets per src: %d\n' % (max_unique_targets))
    result_txt_str += ('Total #unique predictions: %d\n' % num_unique_predictions)

    # report statistics and scores for all predictions and targets
    result_txt_str_all, field_list_all, result_list_all = report_stat_and_scores(num_filtered_predictions,
                                                                                 num_unique_targets, num_src,
                                                                                 score_dict, topk_dict['all'], 'all')
    result_txt_str_present, field_list_present, result_list_present = report_stat_and_scores(
        num_present_filtered_predictions, num_present_unique_targets, num_src, score_dict, topk_dict['present'],
        'present')
    result_txt_str_absent, field_list_absent, result_list_absent = report_stat_and_scores(
        num_absent_filtered_predictions, num_absent_unique_targets, num_src, score_dict, topk_dict['absent'], 'absent')
    result_txt_str += (result_txt_str_all + result_txt_str_present + result_txt_str_absent)
    field_list = field_list_all + field_list_present + field_list_absent
    result_list = result_list_all + result_list_present + result_list_absent

    # Write to files
    results_txt_file = open(pred_fn.replace('.txt', '.result.txt'), "w")

    results_txt_file.write(result_txt_str)
    results_txt_file.close()

    results_tsv_file = open(pred_fn.replace('.txt', '.result.csv'), "w")
    results_tsv_file.write(','.join(field_list) + '\n')
    results_tsv_file.write(','.join('%.5f' % result for result in result_list) + '\n')
    results_tsv_file.close()

    if res_fn:
        with open(res_fn, 'a') as f:
            select_cols = [2, 5, 15, 16, 20, 23, 38, 41, 20, 43]
            f.write(','.join([pred_fn] + ['%.5f' % result_list[i] for i in select_cols]) + '\n')

    if verbose:
        print('\n\nThe full results for %s:' % pred_fn)
        print(result_txt_str)
    else:
        select_cols = [2, 5, 15, 16]
        print('\n' + ','.join([field_list[i] for i in select_cols]))
        print('All: ' + ','.join(['%.5f' % result_list[i] for i in select_cols]))
        present_select_cols = [20, 23]
        print('Present: ' + ','.join(['%.5f' % result_list[i] for i in present_select_cols]))
        absent_select_cols = [38, 41]
        print('Absent: ' + ','.join(['%.5f' % result_list[i] for i in absent_select_cols]))

    # select_cols = [2, 5, 15, 16, 20, 23, 38, 41]
    select_cols = [2]
    interested_metrics = [result_list[i] for i in select_cols]
    print('================================Finish evaluate=================================\n')
    return interested_metrics


# field_list.index('macro_avg_r@5_absent')
# 43
# field_list.index('macro_avg_f1@1_present')
# 20


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', type=str, required=True, help="Path of the prediction file.")
    parser.add_argument('-src', type=str, help="Path of the source file.")
    parser.add_argument('-ocr_fn', type=str, help="Path of the ocr file.")
    parser.add_argument('-res_fn', type=str, default='auto_results_new.csv', help="")
    parser.add_argument('-verbose', type=bool, default=False)
    parser.add_argument('-ocr_only', type=bool, default=False)

    opt = parser.parse_args()
    pred_evaluate(opt.pred, opt.src, opt.res_fn, opt.ocr_only, opt.ocr_fn, opt.verbose)
