import json
import os
import pandas as pd
import pickle
from multiprocessing import Pool


JUDGE = None

task_type = 'anomaly'    # anomaly normal

resource_fields = {'process.runtime.version', 'container.id', 'process.pid', 'node.ip', 'k8s.replicaset.name',
                   'process.runtime.description', 'os.description', 'process.executable.path', 'telemetry.sdk.version',
                   'process.command_line', 'telemetry.auto.version', 'k8s.namespace.name', 'telemetry.sdk.language',
                   'k8s.container.name', 'k8s.node.name', 'host.arch', 'os.type', 'k8s.deployment.name',
                   'process.runtime.name', 'telemetry.sdk.name', 'k8s.pod.name'}

# 删除
delete_fields = {'net.sock.peer.port', 'http.target', 'net.sock.peer.addr', 'thread.name', 'thread.id', 'status',
                 'http.url', 'http.user_agent', 'process.command_line', 'http.client_ip', 'process.pid',
                 'k8s.replicaset.name', 'net.peer.port', 'net.peer.name', 'events'}

# 保留不删除
extra_fields = {'traceId', 'spanId', 'startTimeUnixNano', 'endTimeUnixNano', 'parentSpanId'}

name_kind = {'HTTP GET', 'HTTP DELETE', 'HTTP POST', 'HTTP PUT', 'GET', 'DELETE', 'POST', 'PUT'}


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


class Span:

    def __init__(self, trace_id, span_id, parent_span_id, name, kind, start_time_unix_nano, end_time_unix_nano, status, attrs):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.kind = kind
        self.start_time_unix_nano = start_time_unix_nano
        self.end_time_unix_nano = end_time_unix_nano
        self.status = status
        self.attrs = attrs

    #     self.sons = []

    # def add_son(self, span):
    #     self.sons.append(span)

    # def sort_sons(self):
    #     self.sons = sorted(self.sons, key=lambda span: span.end_time_unix_nano)


class Trace:

    def __init__(self, trace_id, trace_data):
        self.root_span = None
        self.data = None                # 提取得到的数据

        self.trace_id = trace_id
        self.normal_delay = JUDGE

        self.span_num = 0
        self.anomaly = None
        self.root_url = ''

        self.spans = {}    # span_id -> span_obj
        self.sons = {}    # span_id -> [son_span_id, ...]

        self.init_trace(trace_data)
        self.clean_extra_span_attrs()


    def init_trace(self, trace_data):
        has_father = {}  # span_id -> True or False
        root_duration = -1
        for batch in trace_data['batches']:
            batch_resource = batch['resource']
            resouce_attrs = {}
            for item in batch_resource['attributes']:
                resouce_attrs[item['key']] = next(iter(item['value'].values()))

            for ins_span in batch['instrumentationLibrarySpans']:
                for span in ins_span['spans']:
                    self.span_num += 1
                    span_attrs = {}
                    for k, v in span.items():
                        if k != 'attributes':
                            span_attrs[k] = v
                        else:
                            for item in v:
                                span_attrs[item['key']] = next(iter(item['value'].values()))

                    span_attrs.update(resouce_attrs)
                    span_obj = Span(span['traceId'],
                                    span['spanId'],
                                    span['parentSpanId'] if 'parentSpanId' in span else '',
                                    span['name'],
                                    span['kind'],
                                    span['startTimeUnixNano'],
                                    span['endTimeUnixNano'],
                                    span['status'],
                                    span_attrs)
                    self.spans[span['spanId']] = span_obj

                    if 'parentSpanId' in span:
                        parent_span_id = span['parentSpanId']
                        if parent_span_id not in self.sons:
                            self.sons[parent_span_id] = []
                        self.sons[parent_span_id].append(span['spanId'])
                    else:
                        self.root_span = span_obj
                        root_duration = int(span['endTimeUnixNano']) - int(span['startTimeUnixNano'])

        if not self.root_span:
            return

        for span_id, span_obj in self.spans.items():
            if '/sandbox/chaosblade/module/http/*' in span_obj.name:
                self.root_span = None
                return  # chaosblade对应的trace
            if span_obj.attrs['service.name'] == 'test':
                self.root_span = None
                return  # 去掉 test 数据
            # if 'http.url' in span_obj.attrs:    # client span
            #     if span_id not in self.sons:
            #         self.root_span = None
            #         return  # 有http.url但是没有对应的server span
            #     son = self.spans[self.sons[span_id][0]]
            #     # assert len(sons[span_id]) == 1  # otel有例外情况，一个client span有两个server span（大量试验中仅发现了一条）
            #     if len(self.sons[span_id]) > 1:  # 一个 client span 对应多个 server span 则过滤掉
            #         self.root_span = None  # otel自身存在的bug
            #         return
            #     son_http_route = son.attrs['http.route']   # 在 server 里面才有 http.route
            #     assert 'http.route' not in span_obj.attrs
            #     span_obj.attrs['http.route'] = son_http_route
            #     http_url_info = span_obj.attrs['http.url'].split(':')
            #     span_obj.attrs['http.url'] = f"{http_url_info[0]}:{http_url_info[1]}:{http_url_info[2].split('/')[0]}{son_http_route}"

            if 'http.url' in span_obj.attrs:    # client span
                http_route = '/'+'/'.join(span_obj.attrs['http.url'].split(':')[2].split('/')[1:])
                span_obj.attrs['http.route'] = http_route
                if span_id not in self.sons:
                    if task_type == 'normal':
                        self.root_span = None
                        return  # 有http.url但是没有对应的server span
                    else:
                        self.anomaly = True
                elif len(self.sons[span_id]) > 1:  # 一个 client span 对应多个 server span 则过滤掉
                    self.root_span = None  # otel自身存在的bug
                    return

            if span_id in self.sons:
                for son_span_id in self.sons[span_id]:
                    has_father[son_span_id] = True
                    # span_obj.add_son(self.spans[son_span_id])

        no_fa_cnt = 0
        for span_id, span_obj in self.spans.items():
            if span_id not in has_father:
                no_fa_cnt += 1
                if no_fa_cnt > 1:  # 去掉多个根节点的情况
                    self.root_span = None
                    return

        if 'http.method' in self.root_span.attrs and 'http.route' in self.root_span.attrs:
            self.root_url = ' '.join([self.root_span.attrs['http.method'], self.root_span.attrs['http.route']])
            a_key = str((self.root_url, self.span_num))
            if not self.anomaly:
                if a_key in self.normal_delay:
                    self.anomaly = root_duration > self.normal_delay[a_key] + 100_000_000
                else:
                    self.anomaly = root_duration > 1_000_000_000
        else:
            self.root_span = None
            return


    def clean_extra_span_attrs(self):
        # span_attrs_clean
        for span_id, span_obj in self.spans.items():
            new_span_attrs = {}
            for attr_name, attr_value in span_obj.attrs.items():
                # HTTP client span
                if span_obj.kind == 'SPAN_KIND_CLIENT' and span_obj.name in name_kind:
                    if attr_name in resource_fields:
                        continue

                if attr_name in delete_fields:
                    continue

                if attr_name == 'http.method':
                    continue

                if attr_name == 'http.route':
                    new_span_attrs[attr_name] = span_obj.attrs['http.method'] + attr_value
                    continue
                
                new_span_attrs[attr_name] = attr_value
            self.spans[span_id].attrs = new_span_attrs


    def get_trace_graph(self):
        if not self.root_span:
            return None
        
        vertexs = {'0': 'start'}
        edges = {}
        str_set = set()
        trace_duration = {}
        spanIdMap = {'-1': 0}
        spanIdCounter = 1
        rootSpan = None
        spanMap = {}
        spanChildrenMap = {}
        is_abnormal = 0
        chaos_root = ''
        services = []

        # generate span dict
        for span_id, span_obj in self.spans.items():
            if span_obj.parent_span_id == '':
                # root span
                span_obj.attrs.update({'parentSpanId': '-1'})
                span_attrs = span_obj.attrs
            else:
                span_attrs = span_obj.attrs
            spanMap[span_id] = span_attrs
            if span_attrs['parentSpanId'] not in spanChildrenMap.keys():
                spanChildrenMap[span_attrs['parentSpanId']] = []
            spanChildrenMap[span_attrs['parentSpanId']].append(span_attrs)

        # remove internal span
        for span_id, span_obj in self.spans.items():
            if span_obj.kind != 'SPAN_KIND_INTERNAL':
                continue
            if span_obj.parent_span_id == '':
                # root span
                span_obj.attrs.update({'parentSpanId': '-1'})
                span_attrs = span_obj.attrs
            else:
                span_attrs = span_obj.attrs
            if spanMap.get(span_attrs['parentSpanId']) is None:
                return None
            else:
                if span_id in spanChildrenMap.keys():
                    # not leaf node
                    internal_span_children = spanChildrenMap[span_id]
                else:
                    # leaf node
                    internal_span_children = []
                # current parent 节点有可能是 internal，需要删除该节点中子节点的目标 span 的 span id
                internal_span_parent_current = spanMap[span_attrs['parentSpanId']]
                # final parent 节点是向上找最近的一个非 internal 节点，需要在该节点的子节点中，添加目标节点的所有 子节点
                current_span_attrs = span_attrs
                while True:
                    # step1: 获取当前internal span的parent span，即为parent_span_tmp
                    parent_span_attrs_tmp = spanMap[current_span_attrs['parentSpanId']]
                    # step2: 判断如果parent_span_tmp非internal span，则令internal_span_parent_final为parent_span_tmp，并break；
                    #        否则，判断parent_span_tmp是否为root节点，如果是则break；否则令parent_span_tmp的父节点为当前internal span
                    if parent_span_attrs_tmp['kind'] != 'SPAN_KIND_INTERNAL':
                        internal_span_parent_final = parent_span_attrs_tmp
                        break
                    else:
                        if parent_span_attrs_tmp['parentSpanId'] == '-1':
                            return None
                        else:
                            current_span_attrs = parent_span_attrs_tmp
                spanChildrenMap[internal_span_parent_current['spanId']].remove(span_attrs)
                for child in internal_span_children:
                    child['parentSpanId'] = internal_span_parent_final['spanId']
                    spanChildrenMap[internal_span_parent_final['spanId']].append(child)
                # 把当前的internal span从spanChildrenMap中删除
                if spanChildrenMap.get(span_attrs['spanId']):
                    del spanChildrenMap[span_attrs['spanId']]

        # process other spans
        self.spans = dict(sorted(self.spans.items(), key=lambda x: x[1].start_time_unix_nano))
        for span_id, span_obj in self.spans.items():
            if span_obj.kind in ['SPAN_KIND_INTERNAL']:
                continue

            # 为了拿到当前节点的父节点 span id
            def get_clean_info(spanChildrenMap, current_span_id):
                for parent_spanId, child_span_attrs_list in spanChildrenMap.items():
                    for child_span_attrs in child_span_attrs_list:
                        if child_span_attrs['spanId'] == current_span_id:
                            return parent_spanId, child_span_attrs
            
            parentSpanId, span_attrs = get_clean_info(spanChildrenMap, span_id)

            if parentSpanId not in spanIdMap.keys():
                spanIdMap[parentSpanId] = spanIdCounter
                spanIdCounter += 1

            if span_id not in spanIdMap.keys():
                spanIdMap[span_id] = spanIdCounter
                spanIdCounter += 1

            vid, pvid = str(spanIdMap[span_id]), str(spanIdMap[parentSpanId])

            # span id should be unique
            if vid not in vertexs.keys():
                # opname = '/'.join([span_attrs['service'], span_attrs['http.url']])
                # vertexs[vid] = [span_attrs['service'], span_attrs['http.url']]
                vertexs[vid] = span_attrs

            # get edges directed to current span
            if pvid not in edges.keys():
                edges[pvid] = []
            edges[pvid].append(vid)
        
        # anomaly detection
        for span_attrs in vertexs.values():
            if span_attrs == 'start':
                continue
            if 'http.status_code' not in span_attrs.keys():
                continue
            elif int(int(span_attrs['http.status_code'])/100) not in [2, 3]:   # status code 200-299 or 300-399
                self.anomaly = True
                break

        # edges and vertexs check
        vertexs_id_set = set([vertex_id for vertex_id in vertexs.keys() if vertex_id!='start'])
        edges_id_set = set()
        for parent_id, child_ids in edges.items():
            edges_id_set.add(parent_id)
            for child_id in child_ids:
                edges_id_set.add(child_id)
        assert vertexs_id_set == edges_id_set, "some errors occur in edges and vertexs"

        # output graph
        self.data = {
            'root_url': self.root_url,
            'root_delay': int(self.root_span.end_time_unix_nano) - int(self.root_span.start_time_unix_nano),  # 单位：纳秒
            'span_num': self.span_num,
            'anomaly': self.anomaly,
            'vertexs': vertexs,
            'edges': edges
        }
        return self.data


    def get_data(self):
        if not self.root_span:
            return None
        self.get_seq(self.root_span)
        return self.data

    def get_seq(self, span):
        seq = []

        def span_dict(path_event, span):
            span.attrs['path_event'] = path_event
            return dict(span.attrs)

        def dfs(span):
            seq.append(span_dict('span-s', span))

            span.sort_sons()
            select_sons = []
            for i in range(len(span.sons) - 1, -1, -1):
                son = span.sons[i]
                if son.name == 'FilteringWebHandler.handle' or son.kind == 'SPAN_KIND_CONSUMER':
                    continue
                if not select_sons:
                    select_sons.append(son)
                else:
                    last_son = select_sons[-1]
                    last_st = last_son.start_time_unix_nano
                    ed = son.end_time_unix_nano
                    if ed <= last_st:
                        select_sons.append(son)

            for i in range(len(select_sons) - 1, -1, -1):
                son = select_sons[i]
                dfs(son)

            seq.append(span_dict('span-e', span))

        dfs(span)

        self.data = {
            'trace_id': self.trace_id,
            'root_url': self.root_url,
            'root_delay': int(span.end_time_unix_nano) - int(span.start_time_unix_nano),  # 单位：纳秒
            'span_num': self.span_num,
            'anomaly': self.anomaly,
            'critical_path': seq
        }


def work(job):
    input_file = job['input_file']
    output_file = job['output_file']
    print('[input_file]', input_file)
    print('[output_file]', output_file)
    if os.path.exists(output_file):
        print('[skip exists]', output_file)
        return

    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)

    raw_res = [(trace_id, Trace(trace_id, trace_data).get_trace_graph()) for trace_id, trace_data in input_data.items()]
    res = dict([item for item in raw_res if item[1]])
    print(f'write output: {output_file}, num raw_res:{len(raw_res)}, num res:{len(res)}')
    with open(output_file, 'w') as f:
        json.dump(res, f)


def extract_fault(jobs, records_path):
    INPUT_DIR = '/home/zt/data/TrainTicket_traces/exp_raw_data/'
    OUTPUT_DIR = '/home/zt/TraceRAG/dataset/processed_data/trace_graphs/fault_test/'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    records = pd.read_csv(records_path)
    for file in records['filename']:
        jobs.append({
            'input_file': os.path.join(INPUT_DIR, f'{file}.pkl'),
            'output_file': os.path.join(OUTPUT_DIR, f'{file}.json'),
        })


def extract_normal(jobs):
    INPUT_DIR = '/home/zt/data/TrainTicket_traces/normal/'
    OUTPUT_DIR = '/home/zt/TraceRAG/dataset/processed_data/trace_graphs/normal_test/'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file in os.listdir(INPUT_DIR):
        jobs.append({
            'input_file': os.path.join(INPUT_DIR, file),
            'output_file': os.path.join(OUTPUT_DIR, f"{file[:-4]}.json"),
        })


if __name__ == '__main__':
    anomaly_judge_file_path = '/home/zt/TraceRAG/dataset/processed_data/anomaly_judge/normal_all.json'
    inject_info_path = '/home/zt/data/TrainTicket_traces/inject_info_new.csv'
    assert task_type in ('anomaly', 'normal'), f'task not support for {task_type}'
    processes_num = 8

    JUDGE = load_json(anomaly_judge_file_path)

    jobs = []
    if task_type == 'anomaly':
        records_path = inject_info_path
        assert records_path, 'records path can not be empty'
        extract_fault(jobs, records_path)
    if task_type == 'normal':
        extract_normal(jobs)

    jobs = [job for job in jobs if not os.path.exists(job['output_file'])]
    print('jobs count:', len(jobs))

    # test
    for job in jobs:
        work(job)

    # with Pool(processes_num) as p:
    #     p.map(work, jobs)
