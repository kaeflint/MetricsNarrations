import functools
import json
import re
from dataclasses import dataclass
from typing import List

from data_utils import RDFDataSetForTableStructured
from src.data_utils import getClassLabels, roundN
from src.model_utils import setupTokenizer

identicals = {'sensitivity': 'recall', 'true positive rate': 'recall'}


@dataclass
class EvaluationNarrationInstanceDetails:
    metric_names: List
    class_labels: list
    values: List
    rates: List
    dataset_balanced: bool = True

    def cleanOutput(self, narration):
        return [functools.reduce(lambda a, kv: a.replace(*kv), self.class_placeholders.items(),
                                 re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narration]][0]

    def processEvaluationScores(self,):
        """
        This function ties up all the information passed to it into a preamble from which the performance of the classification is 
        narrated
        """
        flag = 'is_imbalanced' if not self.dataset_balanced else 'is_balanced'
        classLabels = getClassLabels(len(self.class_labels))

        # Create dict to keep track of the class labels placeholders
        self.class_placeholders = {pl: cl for pl,
                                   cl in zip(classLabels, self.class_labels)}

        classes_string = ', '.join(classLabels[:-1])+' and '+classLabels[-1]

        b1 = f'ml_task | data_dist | {flag} '
        b2 = f'ml_task | class_labels | {classes_string}'

        task_description = '<TaskDec> ' + b1 + '&& ' + b2 + ' '

        # Process the metrics and their corresponding scores
        assert len(self.metric_names) == len(self.values) and len(self.metric_names) == len(
            self.rates), "Error processing the metrics and their scores"

        metrics_info = []
        metrics_list = []
        values_list = []
        rates_list = []
        for idx, (m, v, r) in enumerate(zip(self.metric_names, self.values, self.rates)):
            mx = m.lower().replace('-score', '').strip()
            mx = m.lower().replace(' score', '').strip()
            mx = m.lower().replace('score', '').strip()
            score_rate = ''
            if int(r) in [4, 5]:
                score_rate = 'HIGH'
            elif int(r) in [3]:
                score_rate = 'MODERATE'
            else:
                score_rate = 'LOW'

            m = m.replace('-', '')
            metric_string = f'{m.lower()} | VALUE_{score_rate} | {roundN(v,2)}%'
            if mx.lower() in identicals.keys():

                metric_string += ' && ' + \
                    f'{m.lower()} | also_known_as | {identicals[mx]}'

            metrics_info.append(metric_string)

            metrics_list.append(m.replace('-', '').lower())
            values_list.append(f'{roundN(v,2)}%')
            rates_list.append(f'{score_rate}')

        metrics_score_string = ' && '.join(metrics_info)+' '

        metricsData = {'metrics': metrics_list,
                       'values': values_list,
                       'rates': rates_list}

        metrics_summary = '<MetricsInfo> '+metrics_score_string

        reps = [metrics_summary, task_description]

        preamble = ' <|section-sep|> '.join(reps) + \
            ' <|section-sep|> <|table2text|> '

        class_dict = {f'C{i+1}': c for i, c in enumerate(classLabels)}
        class_dict.update({f'c{i+1}': c for i, c in enumerate(classLabels)})
        class_dict.update({'F1-score': 'F1score', 'F1-Score': 'F1score',
                           'F2-score': 'F2score', 'F2-Score': 'F2score'})

        self.class_placeholders.update(
            {'F1score': 'F1-score', 'F2score': "F2-score"})

        return {'preamble': preamble, 'classes': classLabels,
                'dataset_attribute': [flag], **metricsData, 'narration': ''}


class NarrationDataSet:
    def __init__(self, modelbase, max_preamble_len=160,
                 max_len_trg=185,
                 max_rate_toks=8,
                 lower_narrations=True,
                 process_target=True,) -> None:

        # Get the tokenizer
        self.tokenizer_ = tokenizer_ = setupTokenizer(modelbase=modelbase)
        self.modelbase = modelbase
        self.max_preamble_len = max_preamble_len
        self.max_len_trg = max_len_trg
        self.max_rate_toks = max_rate_toks
        self.lower_narrations = lower_narrations
        self.process_target = process_target

    def fit(self, trainset, testset):
        self.train_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                          trainset,
                                                          self.modelbase, max_preamble_len=self.max_preamble_len,
                                                          max_len_trg=self.max_len_trg,
                                                          max_rate_toks=self.max_rate_toks,
                                                          lower_narrations=self.lower_narrations,
                                                          process_target=self.process_target,
                                                          use_raw=False)
        self.test_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                         testset,
                                                         self.modelbase, max_preamble_len=self.max_preamble_len,
                                                         max_len_trg=self.max_len_trg,
                                                         max_rate_toks=self.max_rate_toks,
                                                         lower_narrations=self.lower_narrations,
                                                         process_target=self.process_target,
                                                         use_raw=False)

    def transform(self, pack):
        return self.test_dataset.processTableInfo(pack)

    def inferenceTransform(self, pack: list):
        if type(pack) is not list:
            pack = [pack]
        return RDFDataSetForTableStructured(self.tokenizer_,
                                            pack,
                                            self.modelbase, max_preamble_len=self.max_preamble_len,
                                            max_len_trg=self.max_len_trg,
                                            max_rate_toks=self.max_rate_toks,
                                            lower_narrations=self.lower_narrations,
                                            process_target=self.process_target,
                                            use_raw=False)[0]
