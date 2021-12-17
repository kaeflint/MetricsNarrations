import datetime
import functools
import json
import os
import random
import re
import nltk
import numpy as np
import pandas as pd
import sacrebleu
import torch

identicals = {'sensitivity': 'recall', 'true positive rate': 'recall'}
preamble_structure = ['''
On the given  <dataset> dataset, a model was trained to predict <class_string>. Summarize the overall performance 
of the model given that evaluation metrics and their corresponding scores are: <metric_string> as shown in the table: 
''',
                      '''
                    Describe the overall performance of the model trained to predict <class_string> on this <dataset> dataset. The evaluation metrics score were <metric_string> as shown in the table: 
                    ''']
def normalize_text(s):
  # pylint: disable=unnecessary-lambda
  tokenize_fn = lambda x: sacrebleu.tokenizers.Tokenizer13a()(x)
  return tokenize_fn(s.strip().lower())

def writeToFile(content, filename):
    fil = filename+'.txt'
    if os.path.exists(fil):
        os.remove(fil)
    with open(fil, 'x') as fwrite:
        fwrite.writelines("%s\n" % s for s in content)
    print('Done')
    return



def roundN(n, p=1):
    dec, integ = np.modf(n)
    val = integ + np.round(dec, p)
    return val


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def getClassLabels(nb_classes):
    # The class label token is represented as #C{chr(i+97).upper()}
    classes = []
    for i in range(nb_classes):
        cl = '#C'+chr(i+97).upper()
        classes.append(cl)
    return classes


def processMetricsInformationDict(metric_scores, metric_rates, augment=False, identicals={}, nb_metrics=6):
    pp = pd.read_json(json.loads(metric_scores))
    metric_rates = json.loads(metric_rates)
    if augment:
        temp_cols = pp.columns.to_list()
        random.shuffle(temp_cols)
        pp = pp[temp_cols]
    metrics = [s.strip() for s in pp.columns.to_list()]
    metric_rates = {s.strip(): v for s, v in metric_rates.items()}
    values = pp.values.tolist()[0]

    metrics_score_string = '<TM> '

    # Make sure the ratings of all metrics are provided
    #print(metrics, metric_rates.keys())
    assert set(metrics) == set(list(metric_rates.keys()))

    metrics_info = []
    metrics_list = []
    values_list = []
    rates_list = []
    for idx, (m, v) in enumerate(zip(metrics, values)):
        mx = m.lower().replace('-score', '').strip()
        mx = m.lower().replace(' score', '').strip()
        mx = m.lower().replace('score', '').strip()
        score_rate = ''
        if int(metric_rates.get(m, 0)) in [4, 5]:
            score_rate = 'HIGH'
        elif int(metric_rates.get(m, 0)) in [3]:
            score_rate = 'MODERATE'
        else:
            score_rate = 'LOW'

        metrics_list.append(m.replace('-', '').lower())
        values_list.append(f'{roundN(v,2)}%')
        rates_list.append(f'{score_rate}')
    return {'metrics': metrics_list,
            'values': values_list,
            'rates': rates_list}


def normalize_whitespace(string):
    return re.sub(r'(\s)\1{1,}', r'\1', string)

def processMetricsInformation(metric_scores, metric_rates, augment=False, identicals={}):
    pp = pd.read_json(json.loads(metric_scores))
    metric_rates = json.loads(metric_rates)
    if augment:
        temp_cols = pp.columns.to_list()
        random.shuffle(temp_cols)
        pp = pp[temp_cols]
    metrics = [s.strip() for s in pp.columns.to_list()]
    metric_rates = {s.strip():v for s,v in metric_rates.items()}
    values = pp.values.tolist()[0]
    metrics_score_string = '<TM> '

    # Make sure the ratings of all metrics are provided
    assert set(metrics) == set(list(metric_rates.keys()))
    metrics_info = []
    for idx, (m, v) in enumerate(zip(metrics, values)):
        mx = m.lower().replace('-score', '').strip()
        mx = m.lower().replace(' score', '').strip()
        mx = m.lower().replace('score', '').strip()
        score_rate = ''
        if int(metric_rates.get(m, 0)) in [4, 5]:
            score_rate = 'VALUE_HIGH'
        elif int(metric_rates.get(m, 0)) in [3]:
            score_rate = 'VALUE_MODERATE'
        else:
            score_rate = 'VALUE_LOW'
        
        m= m.replace('-','')
        metric_string = f'{m.lower()} | {score_rate} | {roundN(v,2)}%' 
        if mx.lower() in identicals.keys():

            metric_string += ' && ' + \
                f'{m.lower()} | also_known_as | {identicals[mx]}'

        metrics_info.append(metric_string)

    metrics_score_string = ' <|> '.join(metrics_info)+' '

    return '<MetricsInfo> '+metrics_score_string


def processDataSetInformation(flag, datasetInfo):
    model_classes_pattern = re.compile('<b>.*?</b>')
    model_classes_cleanup = re.compile('<b>.*?|</b>')
    classes = [re.sub(model_classes_cleanup, '', s)
               for s in re.findall(model_classes_pattern, datasetInfo,)]
    _mclass = []
    for c in classes:
        if ',' in c:
            c = c.strip().split(',')
            _mclass.extend(c)
        else:
            _mclass.append(c)
    if flag == 1:
        flag = '<|IMBALANCED|>'
    else:
        flag = '<|BALANCED|>'
    b1 = f'ml_task | dataset_attributes | {flag} '

    class_labels = getClassLabels(len(_mclass))
    classes_string = ', '.join(class_labels[:-1])+' and '+class_labels[-1]
    b2 = f'ml_task | class_labels | {classes_string}'

    return '<TaskDec> ' + b1 + '&& ' + b2 + ' '


def parseNarrations(narrations, lower=False):
    narrations = " ".join(narrations.replace(
        '<#>', '').replace("\n", '').strip().split())
    return narrations if not lower else narrations.lower()


def linearizeInput(data, identical_metrics={},
                   augnment_metrics=False,
                   augment_output_order=False,
                   no_narration=False,
                   reverse_output=False
                   ):
    dataset_info = processDataSetInformation(
        data["is_dataset_balanced"], data["dataset_info"])
    metrics_info = processMetricsInformation(
        data["metrics_values"], data["imetric_score_rate"], identicals=identical_metrics, augment=augnment_metrics)
    reps = [metrics_info, dataset_info]
    if augment_output_order:
        random.shuffle(reps)
    narration = normalize_whitespace(parseNarrations(
        data['narration'])) if not no_narration else ''

    if not reverse_output:
        return (' <|section-sep|> '.join(reps)+' <|section-sep|> <|table2text|> ', narration)
    else:
        return (narration + ' <|section-sep|> <|text2table|> ', ' <|section-sep|> '.join(reps))


def composePreambleAndInputs(data, identical_metrics={},
                             augnment_metrics=False,
                             augment_output_order=False,
                             no_narration=False,
                             reverse_output=False,):

    preamble = random.choice(preamble_structure)
    flag, datasetInfo = data["is_dataset_balanced"], data["dataset_info"]
    model_classes_pattern = re.compile('<b>.*?</b>')
    model_classes_cleanup = re.compile('<b>.*?|</b>')
    classes = [re.sub(model_classes_cleanup, '', s)
               for s in re.findall(model_classes_pattern, datasetInfo,)]
    _mclass = []
    for c in classes:
        if ',' in c:
            c = c.strip().split(',')
            _mclass.extend([j.strip() for j in c])
        else:
            _mclass.append(c.strip())
    classes_string = ', '.join(_mclass[:-1])+' and '+_mclass[-1]
    class_labels = getClassLabels(len(_mclass))
    classes_string = ', '.join(class_labels[:-1])+' and '+class_labels[-1]

    if flag == 1:
        flag = '<|IMBALANCED|>'
    else:
        flag = '<|BALANCED|>'

    # Parse the metric information
    metrics_info = processMetricsInformationDict(data["metrics_values"],
                                             data["imetric_score_rate"],
                                             identicals=identical_metrics,
                                             augment=augnment_metrics)
    m_list = metrics_info['metrics']
    v_list = metrics_info['values']
    r_list = metrics_info['rates']

    #metric_string =[random.choice([f'{m} equal to {v}', f'{m} of {v}', f'{m}({v})']) +f' which is rated {r.replace("VALUE_","")}' for m,v,r in zip(m_list,v_list,r_list)]
    metric_string = [random.choice([f'{m}', f'{m}', f'{m}'])
                     for m, v, r in zip(m_list, v_list, r_list)]
    metric_string = ', '.join(metric_string[:-1])+' and '+metric_string[-1]

    preamble_dict = {'<class_string>': classes_string,
                     '<dataset>': flag, '<metric_string>': metric_string}
    preamble = [functools.reduce(lambda a, kv: a.replace(*kv),
                preamble_dict.items(),
                                 re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [preamble]][0]

    preamble, narration = linearizeInput(
        data, identical_metrics=identicals, augnment_metrics=identical_metrics)

    narration = parseNarrations(data['narration']) if not no_narration else ''
    class_dict = {f'C{i+1}': c for i, c in enumerate(class_labels)}
    class_dict.update({f'c{i+1}': c for i, c in enumerate(class_labels)})
    class_dict.update({'F1-score': 'F1score', 'F1-Score': 'F1score',
                      'F2-score': 'F2score', 'F2-Score': 'F2score'})
    extended2 = [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),
                                  re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narration]][0]
    output = {'preamble': preamble, 'classes': class_labels,
              'dataset_attribute': [flag], **metrics_info, 'narration': extended2}
    return output
processInputTableAndNarrations = composePreambleAndInputs


class RDFDataSetForTableStructured(torch.utils.data.Dataset):
    def __init__(self, tokenizer, 
                 data_pack, 
                 modelbase,
                 nb_metrics=6,
                 max_preamble_len=150,
                 max_len_trg=200,
                 max_metric_toks=8,
                 max_val_toks=8,
                 max_rate_toks=8,
                 nb_classes=8,
                 lower_narrations=False,
                 process_target=False):
        super().__init__()
        self.modelbase = modelbase
        self.nb_metrics = nb_metrics
        self.tokenizer = tokenizer
        self.data_pack = data_pack
        self.max_preamble_len = max_preamble_len
        self.max_metric_toks = max_metric_toks
        self.max_val_toks = max_val_toks
        self.max_rate_toks = max_rate_toks
        self.max_len_trg = max_len_trg
        self.nb_classes = nb_classes
        self.lower_narrations = lower_narrations
        self.process_target = process_target
        self.preamble_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                           max_length=self.max_preamble_len,
                                                           padding='max_length',
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           return_tensors='pt')

        self.metrics_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                          max_length=self.max_metric_toks,
                                                          padding='max_length',
                                                          add_special_tokens=True,
                                                          truncation=True,
                                                          return_tensors='pt')
        self.value_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                        max_length=self.max_val_toks,
                                                        padding='max_length',
                                                        add_special_tokens=True,
                                                        truncation=True,
                                                        return_tensors='pt')
        self.rate_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                       max_length=self.max_rate_toks,
                                                       truncation=True,
                                                       padding='max_length',
                                                       add_special_tokens=True,
                                                       return_tensors='pt')

        self.clb_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=False,
                                                      max_length=1,
                                                      truncation=True,
                                                      padding='max_length',
                                                      add_special_tokens=False,
                                                      return_tensors='pt')
        self.di_tokenizer = lambda x: self.tokenizer(x,
                                                     return_attention_mask=False,
                                                     max_length=1,
                                                     truncation=True,
                                                     padding='max_length',
                                                     add_special_tokens=False,
                                                     return_tensors='pt')

    def __len__(self,):
        return len(self.data_pack)

    def processTableInfo(self, data_row):
        data_di = data_row['dataset_attribute']
        data_clb = data_row['classes']
        data_target = data_row['narration']
        data_preamble = data_row['preamble']
        data_values = [v.strip() for v in data_row['values']]
        data_rates = [r.strip() for r in data_row['rates']]
        data_metrics = [m.strip() for m in data_row['metrics']]
        target_encoding = self.tokenizer(data_target, max_length=self.max_len_trg,
                                         padding='max_length',
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors='pt'
                                         )
        labels = target_encoding['input_ids']
        

        # process the class labels
        class_labels = self.clb_tokenizer(data_clb)['input_ids']
        nb_classes = class_labels.shape[0]
        if nb_classes < self.nb_classes:
            pads = torch.zeros((self.nb_classes-nb_classes, 1))
            class_labels = torch.cat(
                [class_labels, pads], dim=0).type(torch.IntTensor)

        # process the dataset information
        data_info = self.di_tokenizer(data_di)['input_ids']

        # process the dataset_info row
        values = self.value_tokenizer(data_values)

        # process the labels row
        rates = self.rate_tokenizer(data_rates)

        # process all the rows on metrics
        metrics = self.metrics_tokenizer(data_metrics)

        metrics_seq = metrics['input_ids']
        metrics_attention = metrics['attention_mask']
        nb_metrics = metrics_seq.shape[0]
        if 'bart' not in self.modelbase:
            if self.process_target:
                labels[labels == 0] = -100
            if nb_metrics < self.nb_metrics:
                pads = torch.zeros(
                    (self.nb_metrics-nb_metrics, self.max_metric_toks))
                metrics_seq = torch.cat(
                    [metrics_seq, pads], dim=0).type(torch.IntTensor)
                metrics_attention = torch.cat(
                    [metrics_attention, pads], dim=0).type(torch.IntTensor)

                val_pad = torch.zeros(
                    (self.nb_metrics-nb_metrics, self.max_val_toks))
                rate_pad = torch.zeros(
                    (self.nb_metrics-nb_metrics, self.max_rate_toks))

                values['input_ids'] = torch.cat(
                    [values['input_ids'], val_pad], dim=0).type(torch.IntTensor)
                values['attention_mask'] = torch.cat(
                    [values['attention_mask'], val_pad], dim=0).type(torch.IntTensor)

                rates['input_ids'] = torch.cat(
                    [rates['input_ids'], rate_pad], dim=0).type(torch.IntTensor)
                rates['attention_mask'] = torch.cat(
                    [rates['attention_mask'], rate_pad], dim=0).type(torch.IntTensor)
        else:
            if nb_metrics < self.nb_metrics:
                pads = torch.ones(
                    (self.nb_metrics-nb_metrics, self.max_metric_toks))
                att_pads = torch.zeros_like(pads)
                metrics_seq = torch.cat(
                    [metrics_seq, pads], dim=0).type(torch.IntTensor)
                metrics_attention = torch.cat(
                    [metrics_attention, att_pads], dim=0).type(torch.IntTensor)

                val_pad = torch.ones(
                    (self.nb_metrics-nb_metrics, self.max_val_toks))
                val_pad_att = torch.zeros_like(val_pad)
                rate_pad = torch.ones(
                    (self.nb_metrics-nb_metrics, self.max_rate_toks))
                rate_pad_att = torch.zeros_like(rate_pad)

                values['input_ids'] = torch.cat(
                    [values['input_ids'], val_pad], dim=0).type(torch.IntTensor)
                values['attention_mask'] = torch.cat(
                    [values['attention_mask'], val_pad_att], dim=0).type(torch.IntTensor)

                rates['input_ids'] = torch.cat(
                    [rates['input_ids'], rate_pad], dim=0).type(torch.IntTensor)
                rates['attention_mask'] = torch.cat(
                    [rates['attention_mask'], rate_pad_att], dim=0).type(torch.IntTensor)

        preamble_encoding = self.preamble_tokenizer(data_preamble)
        preamble_tokens = preamble_encoding['input_ids']
        preamble_attention_mask = preamble_encoding['attention_mask']
        return dict(
            preamble_tokens=preamble_tokens.flatten(),
            preamble_attention_mask=preamble_attention_mask.flatten(),
            class_labels=class_labels.flatten(),
            data_info=data_info.flatten(),
            metrics_seq=metrics_seq.flatten(),
            metrics_attention=metrics_attention.flatten(),
            values=values['input_ids'].flatten(),
            rates=rates['input_ids'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_encoding['attention_mask'].flatten(),
            rate_attention=rates['attention_mask'].flatten(),
            value_attention=values['attention_mask'].flatten()
        )

    def __getitem__(self, idx):
        # print(self.data_pack)
        data_row = self.data_pack[idx]
        return self.processTableInfo(data_row)
