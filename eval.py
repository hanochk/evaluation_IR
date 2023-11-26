import sys
import os
import glob 
from argparse import ArgumentParser
import json
import subprocess
import numpy as np
import pandas as pd 

sys.path.append(os.pardir)

curr_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(curr_dir)
from internal.visual_clues_creator import bb_intersection_over_union
from experiments.nebula3_experiments.vg_eval import VGEvaluation
os.environ["TOKENIZERS_PARALLELISM"] = "false" #huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks

min_iou_pascal_voc = 0.5
evaluator = VGEvaluation()

def flatten(lst): return [x for l in lst for x in l]

def cosine_sim(x: float, y: float):
    return np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))

def sm_similarity(t1: str, t2: str):
    embs = evaluator.smanager.similarity_model.encode([' '.join(t1).lower(), ' '.join(t2).lower()])
    sim = cosine_sim(*embs)
    return sim


# Find matches  
def similarity_matching_many_2_many(gt_dict: dict, det_dict: dict, iou_th: float=0.5):
    retriv_res = dict() # storing best IOU for multiple ovelapped edtections
    
    totoal_false_negatives = 0
    totoal_true_positive = 0
    sm_avg_recall = list()
    
    for obj_gt in gt_dict:
        # print(obj_gt['label'], obj_gt['box'])
       
        max_sm_per_gt = -1
        best_simillar_token = ''
        max_iou_th = -1
        for obj_det in det_dict:
            iou = bb_intersection_over_union(obj_gt['box'], obj_det['box'])
            if  iou > iou_th:
                sm = sm_similarity(tuple([obj_gt['label']]), tuple([obj_det['label']])) # get into the format of ('token',)
                # lbl, max_sm_per_gt = [(obj_det['label'], sm) if sm > max_sm_per_gt else (best_simillar_token, max_sm_per_gt)][0]
# Following COCO, The best matching of few detections vs. 1GT when all IOU>IOU_TH, is the one with highest IOU
                lbl, max_sm_per_gt, max_iou_th = [(obj_det['label'], sm, iou) if iou > max_iou_th else (lbl, max_sm_per_gt, max_iou_th)][0]
                # print("gt: {} gt_id {} det: {} sm: {}". format(obj_gt['label'], obj_gt['object_id'], lbl, max_sm_per_gt))
        # Decide whether FN/TP
        if max_sm_per_gt >= 0:# new TP 
            totoal_true_positive += 1
            sm_avg_recall.append(max_sm_per_gt)
            if obj_gt['label'] in retriv_res:
                retriv_res[obj_gt['label']]['n_gt'] = retriv_res[obj_gt['label']]['n_gt'] + 1
                retriv_res[obj_gt['label']]['token_recall_acm'].append(max_sm_per_gt)
                retriv_res[obj_gt['label']]['best_token'].append(lbl)
            else:                
                retriv_res.update({obj_gt['label']:{'n_gt': 1, 'token_recall_acm': [max_sm_per_gt], 'fn': 0, 'best_token': [lbl]}})
        else: # Mis detection FN are only integer countable 
            sm_avg_recall.append(0) # FN reduces recall by averaging 
            totoal_false_negatives += 1
            # TODO add report
            if obj_gt['label'] in retriv_res:
                retriv_res[obj_gt['label']]['n_gt'] = retriv_res[obj_gt['label']]['n_gt'] + 1
                retriv_res[obj_gt['label']]['fn'] = retriv_res[obj_gt['label']]['fn'] + 1
                retriv_res[obj_gt['label']]['token_recall_acm'].append(0)
            else:                
                retriv_res.update({obj_gt['label']:{'n_gt': 1, 'token_recall_acm': [0], 'fn': 1, 'best_token' :[]}})
    
    # Per token/object FN caused by detection miss (miss detection) total misses related to all positives
    token_stat = {k:{'token_sm_avg_recall' : np.mean(v['token_recall_acm']), 
                    'miss_detection': v['fn']/(v['n_gt']), 'best_token': v.get('best_token', '') , 'n_gt': v['n_gt'] } for (k, v) in retriv_res.items()}
    
    mean_sm_avg_recall = np.mean(sm_avg_recall)
    recall = {'mean_sm_avg_recall' :mean_sm_avg_recall, 'token_stat': token_stat, 
              'total_fn': totoal_false_negatives, 'total_tp': totoal_true_positive}
    
    return recall


def save_per_gt_image_results(det_image_name, args, recall_dict, precision_dict, file):
    csv_file_image = os.path.join(args.result_dir, str(os.path.basename(file).split('.')[0]) + '_' +str(det_image_name) + '.csv')
    recall_dict['token_stat'].update({'mean_sm_avg_recall' :recall_dict['mean_sm_avg_recall'], 
    'mean_sm_avg_precision' : precision_dict['mean_sm_avg_recall'] })

    # precision_dict['mean_sm_avg_precision'] = precision_dict['mean_sm_avg_recall']
    # del(precision_dict['mean_sm_avg_recall'])
    precision_dict['total_fp'] = precision_dict['total_fn'] # changing roles 
    del(precision_dict['total_fn'])

# rewrite keys name 
    precision_dict_new = dict()

    for k, v in precision_dict['token_stat'].items():
        # print(v)
        v['token_sm_avg_precision'] = v['token_sm_avg_recall']
        v['n_detections'] = v['n_gt']
        v.pop('n_gt')
        v.pop('token_sm_avg_recall')
        v['false_positive'] = v['miss_detection']
        v.pop('miss_detection')
        # print(v)
        precision_dict_new[k] = v
 

    df = pd.DataFrame(recall_dict['token_stat'])
    df2 = pd.DataFrame(precision_dict_new)
    df_con = pd.concat([df, df2])
    df_con.to_csv(csv_file_image)
    return df_con

def main(args: list = None):

    parser = ArgumentParser()

    parser.add_argument('--groundtruth-dir', type=str, default=None, metavar='PATH',
                        help=".")

    parser.add_argument('--prediction-dir', type=str, default=None, metavar='PATH',
                        help=".")

    parser.add_argument('--result-dir', type=str, default=None, metavar='PATH',
                        help="if given, all output of the training will be in this folder. "
                             "The exception is the tensorboard logs.")
    
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)    

    csv_file = os.path.join(args.result_dir, 'result_recall_precision.csv')

    filenames_gt = glob.glob(args.groundtruth_dir + '/**/*.json', recursive=True)
    all_weighted_vc_recall  =list()
    all_weighted_vc_precision = list()
    all_total_objects_per_image = list()
    
    all_p_r = list()
    df_per_file_stat_md = pd.DataFrame()
    df_per_file_stat_fp = pd.DataFrame()
    
    for file in filenames_gt:
        gt_file_json = json.load(open(file,'r'))
        gt_file_json = gt_file_json[0]
        image_name = os.path.basename(gt_file_json['image_url']).split('.')[0]
        
        print("GT file {} of image {} {}".format(os.path.basename(file).split('.')[0], image_name, 50*'='))

        # frame_name = full_path_modified + '-' + str(str(frame[0]))
        frame_full_path = subprocess.getoutput('find ' + args.prediction_dir + ' -iname ' + '"*' + image_name + '*"')
        
        if not frame_full_path or os.path.basename(frame_full_path).split('.')[-1] != 'json':
            print("Detection file {} is not exist no analysis will be made ".format(os.path.basename(frame_full_path)))
            continue
        if '\n' in frame_full_path:
            frame_full_path = frame_full_path.split('\n')[0]
        
        # if not (os.path.exists(frame_full_path)):
        #     print("Detection file {} is not exist no analysis will be made ".format(os.path.basename(frame_full_path)))
        #     continue
        
        det_file_json = json.load(open(frame_full_path,'r'))
        if isinstance(det_file_json, list):
            det_file_json = det_file_json[0]
        det_image_name = os.path.basename(det_file_json['image_url']).split('.')[0]
        
        if (image_name != det_image_name):
            print("Detection and GT file name not equal", image_name)
            continue

        
        recall_dict = similarity_matching_many_2_many(gt_dict=gt_file_json['objects'], 
                                   det_dict=det_file_json['objects'], 
                                   iou_th=min_iou_pascal_voc)
        if 1:
            # Precison is recall with substituted arguments
            precision_dict = similarity_matching_many_2_many(gt_dict=det_file_json['objects'], 
                                        det_dict=gt_file_json['objects'],
                                        iou_th=min_iou_pascal_voc)
            
            # precision_dict['mean_sm_avg_precision'] = precision_dict['mean_sm_avg_recall']
            # del(precision_dict['mean_sm_avg_recall'])

        
        all_p_r.append({'det_image_name': det_image_name, 
                        'mean_sm_avg_recall': recall_dict['mean_sm_avg_recall'], 
                        'mean_sm_avg_precision': precision_dict['mean_sm_avg_recall'], 
                        'mean_sm_avg_f1': 2/(1/precision_dict['mean_sm_avg_recall'] + 1/recall_dict['mean_sm_avg_recall'])})
        
        total_objects_per_image = recall_dict['total_fn'] + recall_dict['total_tp']
        all_total_objects_per_image.append(total_objects_per_image)
        print("Image {}: Recall {} Precision {}".format(image_name, recall_dict['mean_sm_avg_recall'], precision_dict['mean_sm_avg_recall']))
        all_weighted_vc_recall.append(recall_dict['mean_sm_avg_recall'] * total_objects_per_image)
        all_weighted_vc_precision.append(precision_dict['mean_sm_avg_recall'] * total_objects_per_image)
        

        df_con = save_per_gt_image_results(det_image_name, args, recall_dict, precision_dict, file)
        # df_con.loc[['miss_detection']]
        # df_con.loc[['false_positive']]        
        
        df_per_file_stat_md = pd.concat([df_per_file_stat_md, df_con.loc[['miss_detection']]])
        df_per_file_stat_fp = pd.concat([df_per_file_stat_fp, df_con.loc[['false_positive']]])
        
# Total recall is a weighted sum 
    all_weighted_vc_recall = np.sum(all_weighted_vc_recall)/np.sum(all_total_objects_per_image)
    all_weighted_vc_precision = np.sum(all_weighted_vc_precision)/np.sum(all_total_objects_per_image)
    all_vc_F1 = 2/(1/all_weighted_vc_recall+1/all_weighted_vc_precision)

    
    df_per_file_stat_md_stat = df_per_file_stat_md.mean(axis=0).sort_values(ascending=False)
    df_per_file_stat_fp_stat = df_per_file_stat_fp.mean(axis=0).sort_values(ascending=False )
    print("Mis-detection from worst decending 1.0 is 100% {}". format(df_per_file_stat_md_stat))
    df_per_file_stat_md_stat.transpose().to_csv(os.path.join(args.result_dir, 'result_md.csv'))
    df_per_file_stat_fp_stat.transpose().to_csv(os.path.join(args.result_dir, 'result_fp.csv'))

    all_p_r.append({'all_weighted_vc_recall': all_weighted_vc_recall, 
                    'all_weighted_vc_precision': all_weighted_vc_precision, 
                    'all_vc_F1': all_vc_F1})

    # all_p_r.append({'all_weighted_vc_recall': all_weighted_vc_recall, 
    #                 'all_weighted_vc_precision': all_weighted_vc_precision, 
    #                 'all_vc_F1': all_vc_F1, 'worst_miss_detected_obj': df_per_file_stat_md_stat,
    #                 'worst_FP_obj': df_per_file_stat_fp_stat})
    
    df_all_p_r = pd.DataFrame(all_p_r)
    
    # df_all_p_r = pd.concat([df_all_p_r, df_per_file_stat_md_stat.transpose()])
    df_all_p_r.to_csv(csv_file, index=False)
    print ("All visual clues : Recall {:3f} Precision {:3f} F1 {:3f}".format( all_weighted_vc_recall, all_weighted_vc_precision, all_vc_F1 ))
    pass 

if __name__ == "__main__":
    
    
    main()  # to be called by outside myModule.main(['arg1', 'arg2', 'arg3'])


"""
Run the following script
Make sure that under the --prediction-dir path there are the updated detection JSONs from the visual clues output
The name of the detection JSON file should be the same as in the ground truth JSON image path

python -u pipelines/tasks/visual_clues/evaluation/eval.py --groundtruth-dir /root/notebooks/vidarts_advanced/pipelines/tasks/visual_clues/evaluation/groundtruth --prediction-dir /root/notebooks/vidarts_advanced/pipelines/tasks/visual_clues/evaluation/detections --result-dir /root/notebooks/vidarts_advanced/pipelines/tasks/visual_clues/evaluation/results

TODO @@HK : for attribute based : compute similarity between tuple-2 and tuple-2
compute P/R per image, sort them per object , FN, FP 


"args": ["--groundtruth-dir", "/root/notebooks/vidarts_advanced/pipelines/tasks/visual_clues/evaluation/groundtruth", 
        "--prediction-dir", "/root/notebooks/vidarts_advanced/pipelines/tasks/visual_clues/evaluation/detections",
        "--result-dir", "/root/notebooks/vidarts_advanced/pipelines/tasks/visual_clues/evaluation/results"],
        
GT images : /datasets_2tb/groundtruth/gt_pictures
http://7.182.10.178:9000/datasets_2tb/gt_cctv_vc/gt_pictures/

Detected JSON : http://7.182.10.178:9000/datasets/media/movies/annotation/default/jsons/

GT JSON : 
http://7.182.10.178:9000/datasets_2tb/gt_cctv_vc/

add best token to er file results 
"""