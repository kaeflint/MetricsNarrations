import argparse
from src.datasethandler import NarrationDataSet
from src.inferenceUtils import PerformanceNarrator
parser = argparse.ArgumentParser(
    description='Arguments for Performance Narration Models.')
parser.add_argument('-mt', '--modeltype', type=str, default='baseline', required=True,
                    help="Specifies the model type: baseline, earlyfusion")

parser.add_argument('-mb', '--modelbase', type=str, default='t5-base', required=True,
                    help="Specifies the model base: t5-base, t5-small or t5-large ")
parser.add_argument('-seed', '--seed', type=int, default=43)
parser.add_argument('-bs', '--batch_size', type=int, default=8)
parser.add_argument('-only_eval', '--only_eval', action="store_true")
parser.add_argument('-sc', '--seed_check', action="store_true")
parser.add_argument('-output_path', '--output_path', type=str, required=True)
parser.add_argument('-sbs', '--sample_bs', action="store_true")


args = parser.parse_args()
# Build the Dictionary
params_dict = vars(args)

# Using the value specified by the modelbase, import the class for the model
if 'bart' in args.modelbase:
    from src.narrations_models import BartNarrationModel as ModelBase
elif 't5' in args.modelbase:
    from src.narrations_models import T5NarrationModel as ModelBase
else:
    raise BaseException("Invalid model base specified. For now only BART and T5 model variants are available")

# Create the object to handle the preprocessing 
narrationdataset = NarrationDataSet(args.modelbase,
                                    max_preamble_len=160,
                                    max_len_trg=185, max_rate_toks=8,
                                    lower_narrations=True, 
                                    process_target=True)


model_generator = ModelBase(
    vocab_size=len(narrationdataset.tokenizer_), model_type=args.modeltype,
    modelbase=args.modelbase)



# Load the models

def main():
    pass


if __name__=="__main__":
    main()






