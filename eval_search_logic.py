import json
from typing import Any

from services import Services, create_services, CreationConfig
from search.logic.params import Params
from search.logic.search_logic import SearchLogic
import os
from argparse import ArgumentParser
import pandas as pd #pip install openpyxl
import numpy as np
import matplotlib.pyplot as plt
import copy
import tqdm
from torchmetrics.functional.retrieval import retrieval_average_precision  #pip install torchmetrics
import torch as nn
from torch import Tensor

def flatten(lst): return [x for l in lst for x in l]

def tested_data_statistics(all_lbls, count, count_sort_ind, uniq_labels, groundtruth_dir, result_dir):
    plt.figure()
    # plt.hist(count[count_sort_ind], bins=100)
    # df.set_index('Family').Battles.plot(kind='bar')
    n_top_labels = 10
    heights = count[count_sort_ind][:n_top_labels]
    bars = uniq_labels[count_sort_ind][:n_top_labels] #['A_long', 'B_long', 'C_long']
    y_pos = range(len(bars))
    plt.bar(y_pos, heights)
    # Rotation of the bars names
    # plt.xticks(y_pos, bars, rotation=90)
    plt.xticks(y_pos, bars, rotation=90, fontsize=5)
    
    plt.title("Ratio: {:.2f}[%] of {} Unique labels out of total of {}" \
        .format(100*len(uniq_labels[count_sort_ind])/len(all_lbls), len(uniq_labels[count_sort_ind]), len(all_lbls)))

    plt.savefig(os.path.join(result_dir, str(os.path.basename(groundtruth_dir).split('.')[0]) + '.jpg'), dpi=300)
    plt.close()
    pass

def precision_at_k(y_true: list, y_pred: list, k=12):
    
    """ 
    The ordering here is what matters (https://www.educative.io/answers/what-is-the-mean-average-precision-in-information-retrieval)
    hence as long as you retrive more non relevant documnets the precision goes down despite the fact that relevancy is r(k)=0. Since by the time 
    you reach relevant one the Precsion would be very low. Stil it is ordering dependant. 
    Hence if the expected retrived documents are N but only M<N retrived the N-M non retrived has relevancy=1 with precision=0 
    but averaged by RD (denum) whic equals to all docs should be retrived
    
    the relevant documents not retrieved get a precision score of zero. : https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
    
    
    Computes Precision at k for one sample
    https://www.kaggle.com/code/debarshichanda/understanding-mean-average-precision
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k

def rel_at_k(y_true: list, y_pred: list, k=12):
    """ Computes Relevance at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0
    

def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    for i in range(1, k+1):
        if i <= len(y_pred):# for retrived documents
            ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
        # else: #relevant documents not retrieved get a precision score of zero. NIST : Per the metric description in m_P.c of trec_eval 9.0 
        # https://stackoverflow.com/questions/46374405/precision-at-k-when-fewer-than-k-documents-are-retrieved
        #     ap +=0
        # RD is the number of relevant documents for the query, namely documents not retrived will reduce the avg
    return ap / min(k, len(y_true))

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--groundtruth-dir', type=str, default=None, metavar='PATH',
                        help=".")

    # parser.add_argument('--top-k-retrival', type=str, default=None, metavar='PATH',
    #                     help=".")

    parser.add_argument('--n-labels-search', type=int, default=3, metavar='PATH',
                        help=".")

    parser.add_argument('--result-dir', type=str, default=None, metavar='PATH',
                        help="if given, all output of the training will be in this folder. "
                             "The exception is the tensorboard logs.")
    
    args = parser.parse_args()

    return args

class SearchEngine():
    def __init__(self, algorithm_type='Semantic_FAISS_n_CSS'):
        self.filter_movies_list = []
        #embedding_type = 'blip_on_image'
        self.sub_algorithm_type = 'semantic_labels'
        #self.algorithm_type = 'Temporal_CSS'
        self.algorithm_type = algorithm_type

        # self.services_config_path = "services/creator/config_services_test.json"
        self.services_config_path = "services/creator/config_services_prod.json"

        with open(self.services_config_path, "r") as read_file:
            services_config_as_json = json.load(read_file)
        services_creation_config: CreationConfig = CreationConfig.from_json(services_config_as_json)
        services: Services = create_services(services_creation_config)

        search_config_path = "search/backend/config.json"
        with open(search_config_path, "r") as read_file:
            search_config_as_json = json.load(read_file)

        search_params: Params = Params.from_json(search_config_as_json)

        self.search_logic = SearchLogic(services, search_params)
        # top_k_results = 1 # Top-K 
        self.num_result_per_item = 1
        self.items_types_list = []
        # num_result_per_item = 1 # For Video  how many results per movie ? 
        self.diff_comb = True
        self.gap_step = 0
        
    def __call__(self, search_list, search_tag, top_k_results=1): 
        outputs = self.search_logic.get_search_results(search_list=[search_list], search_tag=search_tag, 
                                                       filtered_movies_list=self.filter_movies_list, 
                                                        items_types_list=self.items_types_list, 
                                                        num_result_per_item=self.num_result_per_item,
                                                        algorithm_type=self.algorithm_type, 
                                                        sub_algorithm_type=self.sub_algorithm_type, 
                                                        top_scores=top_k_results, 
                                                        diff_comb=self.diff_comb,
                                                        gap_step=self.gap_step)

        return outputs
        
    
class Executor():
    def __init__(self, search_engine_obj, search_algo_type):
        self.search_algo_type = search_algo_type

        self.search_engine_obj = search_engine_obj(algorithm_type=self.search_algo_type)
        # self.search_engine_obj.__init__()
        # self.gt_qr_table = gt_qr_table
        
    
    def __call__(self, query_text, ds_tag, top_k_results):
        return self.search_engine_obj(search_list=query_text, search_tag=ds_tag, top_k_results=top_k_results)
        

def eval_search():
    print("This evaluator calculates mAP for Information retrival (AP over all top-k)")
    print("Load user arguments")
    args = parse_args()
    
    search_engine = SearchEngine()
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)    

    df_gt_file = pd.read_excel(args.groundtruth_dir)

    df_gt_file = pd.read_excel(args.groundtruth_dir,names=['gt_image'] + ['label_{}'.format(x+1) for x in range(int(len(df_gt_file.keys())-1))] )
    # Strip labels
    df_gt_file[df_gt_file.columns[1:]] = df_gt_file[df_gt_file.columns[1:]].apply(lambda x: x.str.strip())
    
    all_lbls = flatten([df_gt_file[col].values for col in df_gt_file.columns[1:]]) 
    # REmove nan  
    all_lbls = [x for x in all_lbls if str(x) != 'nan']
    [print("Labels contains asterick!! they were removed ") if any([x for x in all_lbls if "," in str(x)]) else None]
    # Remove asterick in case labels are not seperated well 
    all_lbls = [x for x in all_lbls if "," not in str(x)]
    
    uniq_labels, count = np.unique(all_lbls, return_counts=True)
    count_sort_ind = np.argsort(-count)
    uniq_cnt_sorted = count[count_sort_ind]
    uniq_labels_sorted = uniq_labels[count_sort_ind]
    
    print("Ratio: {:.2f}[%] of {} Unique labels out of total of {}" .format(100*len(uniq_labels[count_sort_ind])/len(all_lbls), len(uniq_labels[count_sort_ind]), len(all_lbls)))
    df_gt_file['gt_image_no_ext'] = df_gt_file['gt_image'].apply(lambda x: x.split('.')[0])
    
    tested_data_statistics(all_lbls, count, count_sort_ind, 
                           uniq_labels, groundtruth_dir=args.groundtruth_dir, result_dir=args.result_dir)

    
    csv_file = os.path.join(args.result_dir, 'result_recall_precision_one_token.csv')

# IDF
    df_queries_label = copy.deepcopy(df_gt_file)
    df_queries_label = df_queries_label.drop(columns=['gt_image', 'gt_image_no_ext'], axis=1)
    # for search_token in uniq_labels_sorted:
    n_docs = len(df_queries_label)
    idf_dict = dict()
    for cnt, label in zip(uniq_cnt_sorted , uniq_labels_sorted): # since each label can 
        idf_dict[label] = np.log10(n_docs/cnt)
    
    top_k_results = n_docs #all_top_k[0]
    total_queries = 0
    total_queries_dict = dict()
    
    query_len  = 1 #in np.arange(1 ,1+ min(len(df_gt_file.keys())-1, args.n_labels_search)):
    all_query_ap = list()
    df_all_ap = list()

    for ix, search_token in enumerate(tqdm.tqdm(uniq_labels_sorted.tolist())):
        all_ap_at_k = list()
        # second_label_cand = [df_queries_label[df_gt_file['gt_image_no_ext'] == x].dropna(axis=1).values for x in  y_true]

        total_queries += 1
        search_list = [search_token]
        top_k_retrival_relevant_to_query = uniq_cnt_sorted[ix]
        outputs = search_engine(search_list=search_list, top_k_results=top_k_results)
        # outputs = search_logic.get_search_results(search_list, self.filter_movies_list, self.items_types_list, num_result_per_item,
        #                                         self.algorithm_type, self.filter_movies_list, top_k_results, self.diff_comb,
        #                                         self.gap_step)
        #  it is ordering dependant. Hence if the expected retrived documents are N but only M<N retrived the N-M non retrived has relevancy=1 with decreasing precision
        y_pred = outputs['movie_name']
           

        y_true = df_gt_file[df_gt_file.isin([search_token]).any(axis=1)]['gt_image_no_ext'].to_list()
        if 0:
            df = [pd.concat([df_gt_file[df_gt_file['gt_image_no_ext'] == x], df]) for x in  y_true]
            df = [df_gt_file[df_gt_file['gt_image_no_ext'] == x].dropna(axis=1) for x in  y_true]
            df.to_csv(os.path.join(args.result_dir, 'unitest_map.csv'), index=False)

        # Per top-K
        topk = len(outputs['movie_name'])
        if topk == 0:
            ap_per_query = 0
        else:
            ap_per_query = average_precision_at_k(y_true=y_true, y_pred=y_pred, 
                                            k=topk)
            
        if len(outputs['movie_name']) <= top_k_retrival_relevant_to_query:
            print("No of retrived {} images are less than GT topk {}".format( len(outputs['movie_name']), 
                                                                              top_k_retrival_relevant_to_query))
            relevant_doc_not_ret = top_k_retrival_relevant_to_query - len(outputs['movie_name'])
            all_ap_at_k.extend([0]*relevant_doc_not_ret) # ultimately the non relevant retrived docs will reduce AP@k
        else:
            # rel(k)=0 anyway
            print("No of retrived {} images are more than GT topk {} label: {} ".format( len(outputs['movie_name']), 
                                                                              top_k_retrival_relevant_to_query, 
                                                                              search_token))

        # ap_at_k = np.mean(all_ap_at_k)
        df_all_ap.append({'token': search_token, 'ap_per_query': ap_per_query, 
                               'retrived_images': len(outputs['movie_name']), 'gt_images': top_k_retrival_relevant_to_query})
        
        all_query_ap.append(ap_per_query)
        if ix%10 == 0:
            df_res = pd.DataFrame(df_all_ap)
            df_res.to_csv(csv_file, index=False)
             
        
    map = np.mean(all_query_ap)
    df_all_ap.append({'map': map})
    df_res = pd.DataFrame(df_all_ap)
    df_res.to_csv(csv_file, index=False)
 
    return map

def compute_ap(relevance_bitmap: list[bool], preds_likelihood=nn.empty):
    """Compute average precision (for information retrieval), as explained in `IR Average precision`_.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        a single-value tensor with the average precision (AP) of the predictions ``preds`` w.r.t. the labels ``target``.

    """
    
    # Likelihood proxy as assumed ordering telling the likelihood implicitly 
    if preds_likelihood == nn.empty: # dummy confidence 
        preds_likelihood = nn.flip(nn.arange(0.0, len(relevance_bitmap)), dims=(0, ))/len(relevance_bitmap)  
    # https://github.com/Lightning-AI/torchmetrics/blob/v1.1.0/src/torchmetrics/functional/retrieval/average_precision.py#L22
    ap = retrieval_average_precision(preds=preds_likelihood, target=Tensor(relevance_bitmap))

    return ap.numpy()

def unittest_ap():
    # https://www.educative.io/answers/what-is-the-mean-average-precision-in-information-retrieval
    rel_k = [True, True ,False, True, False] #pr=[1, 1, 2/3, 3/4, 3/5] AP = sum(pr*rel_k)/np.sum(np.array(rel_k).astype('int')) 
    ap = compute_ap(rel_k)
    assert(np.isclose(ap, 0.9166667))
    print("Pass test No. 1")

    rel_k = [True, True , True, False, False]
    ap = compute_ap(rel_k)
    assert(np.isclose(ap, 1.0))

    print("Pass test No. 2")

    rel_k = [True, False , False, False, True] #pr=[1, 0.5, 1/3, 1/4, 2/5] AP = sum(pr*rel_k)/np.sum(np.array(rel_k).astype('int')) = 0.7
    ap = compute_ap(rel_k)
    assert(np.isclose(ap, 0.7))
    
    print("unittest passed")
        
"""
class MyTestCase(unittest.TestCase):

    def test_bit_exactness_throughput_mem(self):
        Point = namedtuple('Point', ['x', 'y'])
        gpu_id = 0
# Check if GPU exist this test fails if GPU is busy hence
        if check_if_gpu_exist:
            is_cuda_gpu_available = check_if_gpu()
            if is_cuda_gpu_available:
                print("CUDA GPU found! {}".format(is_cuda_gpu_available))
            self.assertTrue(is_cuda_gpu_available)

"""
def unitest_high_level():
    print("This evaluator calculates mAP for Information retrival (AP over all top-k)")
    print("Load user arguments")
    args = parse_args()
    df_gt_file = pd.read_csv(args.groundtruth_dir)
    df_gt_file['gt_image_no_ext'] = df_gt_file['gt_image'].apply(lambda x: x.split('.')[0])

    # df_gt_file = pd.read_csv(args.groundtruth_dir,names=['gt_image'] + ['label_{}'.format(x+1) for x in range(int(len(df_gt_file.keys())-1))] )
    # Strip labels
    # df_gt_file[df_gt_file.columns[1:]] = df_gt_file[df_gt_file.columns[1:]].apply(lambda x: x.str.strip())
    search_token = 'The sky is clear'

    y_true = df_gt_file[df_gt_file.isin([search_token]).any(axis=1)]['gt_image_no_ext'].to_list()
    
    
    # rel_k = [True if y_p == y_t else False for (y_p, y_t) in zip (y_pred, y_true)]
    
    for n_ret in range(1, len(y_true)+1):
        y_pred = y_true[:n_ret]    
        ap_per_query = average_precision_at_k(y_true=y_true, y_pred=y_pred, 
                                            k=len(y_true))
        rel_k = [True if y_p == y_t else False for (y_p, y_t) in zip (y_pred, y_true)]
        # pad the non relevant by 0
        [rel_k if len(rel_k) == len(y_true) else rel_k.extend([False]*(len(y_true)-len(rel_k)))]
        ap = compute_ap(rel_k)
        # preds_likelihood = nn.ones(len(y_true))
        # https://github.com/Lightning-AI/torchmetrics/blob/v1.1.0/src/torchmetrics/functional/retrieval/average_precision.py#L22
        # ap = retrieval_average_precision(preds=preds_likelihood, target=Tensor(rel_k))
        print("AP {} for retrieved {} image out of {} Torch :{}".format(ap_per_query, n_ret, len(y_true), ap))
    
    

    pass

def executor_run(search_engine_obj, gt_qr_table, query_table, ds_tag, 
                 query_tag='exact', test_tag: str='test1', search_algo_type:str ='Semantic_FAISS_n_CSS'):
    
    executor = Executor(search_engine_obj=search_engine_obj, 
                        search_algo_type=search_algo_type) 
                        # gt_qr_table=gt_qr_table
    
    query_ids = pd.unique(gt_qr_table[gt_qr_table[gt_qr_table['ds_tag'] == ds_tag]['query_tag'] == query_tag]['query_id'])
    max_top_k_per_tag = len(gt_qr_table[gt_qr_table[gt_qr_table['ds_tag'] == ds_tag]['query_tag'] == query_tag])
    res_all = list()
    for query_id in query_ids:
        query_text = query_table[query_table['query_id'] == query_id]['query_text'].iloc[0]
        search_res_dict = executor(query_text=query_text, ds_tag=ds_tag, top_k_results=max_top_k_per_tag)
        if search_res_dict:
            url, score = search_res_dict['movie_MDF_url'], search_res_dict['score']
        else:
            url = ''
            score = 0
            
        Search_results_url_score = [(url, score) for url, score in zip(url, score)]
        res_all.append({'test_tag': test_tag, 'query_id': query_id, 
                        'search_results': Search_results_url_score, 'ds_tag': ds_tag, 
                        'search_tag': search_algo_type})
    
    # res_all_df = pd.DataFrame(res_all)
    # res_all_df.to_csv('')
    """
Search Results Table(s): The Executor creates this table every time it evaluates a search algorithm. Structure:
TestTag: str      # The name of the specific test. Given to Executor 
Query: QueryID
SearchResults: List[(URL, score)]      # URL => Is the .
DSTag: str       Why data source tag is needed? 
SearchTag: str       # The search algorithm used.
    
    """
    # res_list = list()
    # res_list.append({'url':'url1', 'ds_tag' :'2k_091123', 'query_tag': 'exact', 'image_label':'1'})
    search_result_table_df = pd.DataFrame(res_all)

    return search_result_table_df

def evaluator_run(gt_qr_table, search_result_table, test_tag):
    ds_tag = search_result_table[search_result_table['test_tag'] == test_tag]['ds_tag']
    max_top_k_per_url_dstag = len(gt_qr_table[gt_qr_table['ds_tag'] == ds_tag.values[0]])
    
    query_ids = pd.unique(search_result_table[search_result_table['test_tag'] == test_tag]['query_id'])
    
    res_all = list()
    for query_id in query_ids:
        relevance_per_query = list()
        fn_per_query = 0
        fp_per_query = 0
        retrived_urls = search_result_table[search_result_table[search_result_table['test_tag'] == test_tag]['query_id'] == query_id]['search_results'].values
        gt_ir_ds_query = gt_qr_table[gt_qr_table[gt_qr_table['ds_tag'] == ds_tag.values[0]]['query_id'] == query_id]
        if isinstance (retrived_urls, np.ndarray):
            retrived_urls = retrived_urls[0]

        for url_n_score in retrived_urls:
            url = url_n_score[0]
            score  = url_n_score[1]
            if any([url[0] in x for x in gt_ir_ds_query['url']]):
                relevance_per_query.append(True)
            else:
                relevance_per_query.append(False)
                fp_per_query += 1
        
        if len(relevance_per_query) < len(gt_ir_ds_query):
            relevance_per_query.extend([False]*(len(gt_ir_ds_query) - len(relevance_per_query)))
            
        ap_query = compute_ap(relevance_per_query) 
# Compute FN        
        for gt_url in gt_ir_ds_query['url']:
            if not any([gt_url in x[0] for x in retrived_urls]):
                fn_per_query += 1
            
        res_all.append({'test_tag': test_tag, 'query_id': query_id, 
                        'ap': ap_query, 'FN_k': fn_per_query, 'FP_k': fp_per_query,
                        'search_tag': search_result_table['search_tag'].values[0]})
            
    eval_table_df = pd.DataFrame(res_all)

    return eval_table_df
"""
Eval Results Table(s): Holds the Average Precision results for a given experiment (ground truth/search algorithm). Structure is:
TestTag: str
GTQRTag: str
QueryTag: str
SearchTag: str
Query: QueryID
AveragePrecision: float    # A number between 0 and 1
FP
FN

"""



def executor_evaluator_wrapper():
    args = parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)    

    # Mockup 
    # gt_qr_table = pd.DataFrame(columns=['url', 'ds_tag', 'query_tag', 'image_label'])
    gtqr_list = list()
    gtqr_list.append({'url':'http://7.182.10.178:9000//datasets/media/movies/201903_00003.jpg', 
                      'ds_tag' :'2k_091123', 'query_tag': 'exact', 'image_label':'1', 'query_id': 1}) #The sky is clear
    gtqr_list.append({'url':'http://7.182.10.178:9000//datasets/media/movies/shenzhenditie_001_2020-07-08-18-18-36_2020-07-08-18-48-38_41300.jpg', 
                      'ds_tag' :'2k_091123', 'query_tag': 'exact', 'image_label':'1', 'query_id': 1}) #The sky is clear
    gtqr_list.append({'url':'http://7.182.10.178:9000//datasets/media/movies/suzhou0910_0_200_frame.jpg', 
                      'ds_tag' :'2k_091123', 'query_tag': 'exact', 'image_label':'1', 'query_id': 1}) #The sky is clear
    gtqr_list.append({'url':'http://7.182.10.178:9000//datasets/media/movies/suzhou0911_2_25_frame_BlendAlphaVerticalLinearGradient.jpg', 
                      'ds_tag' :'2k_091123', 'query_tag': 'exact', 'image_label':'1', 'query_id': 1}) #The sky is clear
    gtqr_list.append({'url':'http://7.182.10.178:9000//datasets/media/movies/015_rtsp.jpg', 
                      'ds_tag' :'2k_091123', 'query_tag': 'exact', 'image_label':'1', 'query_id': 1}) #The sky is clear
    
    gt_qr_table_df = pd.DataFrame(gtqr_list)
    query_table_df = pd.DataFrame([{'query_id': 1, 'query_text': 'The sky is clear'}])
    
    search_result_table_df = executor_run(search_engine_obj=SearchEngine, 
                                       gt_qr_table=gt_qr_table_df,
                                       query_table=query_table_df,
                                       ds_tag = '2k_091123', 
                                       query_tag = 'exact', 
                                       test_tag = 'vis_clues_rev1')

    csv_file = os.path.join(args.result_dir, 'executor_table_results.csv')
    search_result_table_df.to_csv(csv_file, index=False)
    
    eval_result_table_df = evaluator_run(gt_qr_table=gt_qr_table_df, 
                                         search_result_table=search_result_table_df,
                                         test_tag = 'vis_clues_rev1')

    csv_file = os.path.join(args.result_dir, 'evaluator_table_results.csv')
    eval_result_table_df.to_csv(csv_file, index=False)

    return eval_result_table_df

if __name__ == "__main__":
    """
GTQR Table(s): Holds all instances of the search task ground truths. The structure is:
Source: QueryID
Target: URL
DSTag: str	            # Given to Label Generator, saved in CSS and frame_analysis. Represents Data Source.
QueryTag: str             #  e.g “exact”, or “synonyms+Boolean”. Given to Query Generator.
Option<List[Label]>: Image Label     # An optional link back to the actual Image Label the query matches.
   
vehicle_park_20200824_23_-000000003.jpg
shenzhenditie_001_2020-07-08-18-18-36_2020-07-08-18-48-38_41300.jpg
suzhou0910_0_200_frame.jpg
suzhou0911_2_25_frame_BlendAlphaVerticalLinearGradient.jpg
015_rtsp.jpg
035_rtsp.jpg
042_rtsp.jpg



    """
    executor_evaluator_wrapper()
    unittest_ap()
    # eval_search()


"""
 python -u search/testers/eval_search_logic.py --groundtruth-dir /root/notebooks/vidarts_advanced/search/testers/validation/data/test23/CCTV-HQ-2K_HQ-labels-clean.xlsx --result-dir /root/notebooks/vidarts_advanced/search/testers/validation/results/test23

            "args": ["--groundtruth-dir", "/root/notebooks/vidarts_advanced/search/testers/validation/data/test23/CCTV-HQ-2K_HQ-labels-clean.xlsx", 
                    "--result-dir", "/root/notebooks/vidarts_advanced/search/testers/validation/results/test23"],
prod/dev
services_config_path = "services/creator/config_services_dev.json"
services_config_path = "services/creator/config_services_prod.json"
"""