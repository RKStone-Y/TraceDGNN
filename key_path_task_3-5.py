'''
task 3-5: {trace A} 中的最慢调用路径（关键路径）
'''

import json
import yaml
import random
import copy
from tqdm import tqdm

client_name_kind = {'HTTP GET', 'HTTP DELETE', 'HTTP POST', 'HTTP PUT', 'GET', 'DELETE', 'POST', 'PUT'}
span_kinds = {'SPAN_KIND_CLIENT', 'SPAN_KIND_SERVER', 'SPAN_KIND_PRODUCER', 'SPAN_KIND_CONSUMER'}

def get_seq(trace, span_idx, results):
    def span_dict(path_event, span_idx):
        span = copy.deepcopy(trace['vertexs'][span_idx])
        span['path_event'] = path_event
        return dict(span)
    
    def sort_sons(span_idx):
        if span_idx not in trace['edges']:
            return []
        son_idxs = trace['edges'][span_idx]
        sons = [(span_id, span) for span_id, span in trace['vertexs'].items() if span_id in son_idxs]
        sorted_sons = sorted(sons, key=lambda span: span[1]['endTimeUnixNano'])
        return sorted_sons

    def dfs(span_idx):
        results.append(span_dict('span-s', span_idx))

        sorted_sons = sort_sons(span_idx)
        select_sons = []
        for i in range(len(sorted_sons) - 1, -1, -1):
            son = sorted_sons[i]
            if son[1]['name'] == 'FilteringWebHandler.handle' or son[1]['kind'] == 'SPAN_KIND_CONSUMER':
                continue
            if not select_sons:
                select_sons.append(son)
            else:
                last_son = select_sons[-1]
                last_st = last_son[1]['startTimeUnixNano']
                ed = son[1]['endTimeUnixNano']
                if ed <= last_st:
                    select_sons.append(son)

        for i in range(len(select_sons) - 1, -1, -1):
            son = select_sons[i]
            dfs(son[0])

        results.append(span_dict('span-e', span_idx))

    dfs(span_idx)

    return results
            

def generate_task_instance(trace):
    '''
    input
    1. trace: dict, the trace content
    output
    1. answer: list, spanid lists
    '''
    # step1: 获取关键路径序列
    seq_result = list()
    rootspan_idx = trace['edges']['0'][0]
    seq_result = get_seq(trace=trace, span_idx=rootspan_idx, results=seq_result)
    # step2: 获取最慢调用路径的 spanid 序列
    answer = list()
    for span_event in seq_result:
        answer.append(span_event['spanId'])
    
    return list(answer)


if __name__ == "__main__":
    # with open(f'./config.yaml', 'r') as f:
    #     configs = yaml.safe_load(f)
    
    test_file = '/data/TraceRAG/db_abnormal/preprocessed/0.json'
    with open(test_file, 'r') as file:
        all_abnormal_traces = json.load(file)
    
    for trace_id, trace_content in tqdm(all_abnormal_traces.items()):
#         if len(trace_content['vertexs']) >= 10:
#             continue
#         if trace_content['root_url'] in ['DELETE /api/v1/adminrouteservice/adminroute/{routeId}', 'GET /api/v1/adminrouteservice/adminroute', 'POST /api/v1/travelservice/trips/left', 
# 'POST /api/v1/travelservice/trips/left_parallel', 'POST /api/v1/foodservice/createOrderBatch']:
#             continue
        answer = generate_task_instance(trace=trace_content)
        print(answer)
        break
    print('done')