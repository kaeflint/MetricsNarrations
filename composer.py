import argparse
import pickle as pk
import time
from re import S

from pytorch_lightning import seed_everything
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from src.data_utils import *
from src.model_utils import setupTokenizer
parser = argparse.ArgumentParser(
    description='Arguments for Performance Narration Models.')
parser.add_argument('-mt', '--modeltype', type=str, default='baseline', required=True,
                    help="Specifies the model type: baseline, earlyfusion")
parser.add_argument('-mb', '--modelbase', type=str, default='t5-base', required=True,
                    help="Specifies the model base: t5-base, t5-small or t5-large ")
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('-bs', '--batch_size', type=int, default=8)
parser.add_argument('-epochs', '--epochs', type=int, default=20)
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
parser.add_argument('-ws', '--warmup_steps', type=int, default=0)
parser.add_argument('-bottom_k', '--bottom_k', type=int, default=11)
parser.add_argument('-ga', '--gradient_accumulation_steps',
                    type=int, default=10)
parser.add_argument('-only_eval', '--only_eval', action="store_true")
parser.add_argument('-output_path', '--output_path', type=str, required=True)
parser.add_argument('-use_raw','--use_raw', action="store_true")


args = parser.parse_args()
# Build the Dictionary
params_dict = vars(args)
seed_everything(args.seed)

# Setup the hyperparameters
batch_size = args.batch_size
epochs = args.epochs
learning_rate = float(args.learning_rate)
warmup_steps = args.warmup_steps
epsilon = 1e-8
tokenizer = tokenizer_ = setupTokenizer(modelbase=args.modelbase)

device = torch.device("cuda")
print(f'Learning rate is {learning_rate}')


mle_only = True
accumulation_steps = args.gradient_accumulation_steps
processed = pk.load(open('dataset/train_dataset_full.dat', 'rb'))
print(len(processed))

test_data = json.load(open('dataset/test set.json'))
test_sample = []
eval_tables = []
for pc in test_data:
    test_sample.append(processInputTableAndNarrations(
        pc, identical_metrics=identicals))
    # eval_tables.append(parseTableStructureForEval(pc,identicals))
rtest_sample = []
reval_tables = []
for pc in test_data:
    rtest_sample.append(processInputTableAndNarrations(
        pc, identical_metrics=identicals))



dataset = RDFDataSetForTableStructured(tokenizer_,  processed, args.modelbase,max_preamble_len=160,
                                       max_len_trg=180, max_rate_toks=8,
                                       lower_narrations=True, process_target=True,use_raw=args.use_raw)
test_dataset = RDFDataSetForTableStructured(tokenizer_, test_sample, args.modelbase,
                                            max_preamble_len=150, max_rate_toks=1, max_len_trg=180,use_raw=args.use_raw)
# Split into training and validation sets
train_size = int(len(dataset))
val_size = int(len(test_dataset))


train_dataset, val_dataset = dataset, test_dataset

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset)  # Select batches randomly
    , batch_size=batch_size  # Trains with this batch size.
)
# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=1  # Evaluate with this batch size.
)

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))