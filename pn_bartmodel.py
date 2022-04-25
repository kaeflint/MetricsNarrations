import argparse
import pickle as pk
import time
from re import S

from pytorch_lightning import seed_everything
from torch.nn import functional as F


from composer import *
from src.model_utils import setupTokenizer
from src.narrations_models import BartNarrationModel
from src.losses import computeLoss



model_generator = BartNarrationModel(
    vocab_size=len(tokenizer_), model_type=args.modeltype,
    modelbase=args.modelbase)

# compile the model setting up the optimizer and the learning rate schedule
total_steps = len(train_dataloader) * epochs
print(f'Ws: {warmup_steps}')
if warmup_steps ==0:
    warmup_steps =  int(total_steps*float(args.warmup_ratio))
    print(f'Ws {warmup_steps}')
model_generator.compile(
    lr=learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,)


hh= args.modelbase.split('/')[1]
output_path = 'TrainedNarrators/P-NarrationsModels/' + \
    args.modeltype+args.output_path+'/'+hh+'/'#+'/wr'+str(args.warmup_ratio)+'/'

if args.seed_check:
    output_path= output_path+f'/{args.seed}/'
try:
    os.makedirs(output_path)
except:
    pass
print(f'Results will be saved @: {output_path}')
def baselineTraining(step, batch):
    preamble_tokens = batch['preamble_tokens'].to(device)
    preamble_attention_mask = batch['preamble_attention_mask'].to(device)
    labels = batch['labels'].to(device)
    #labels[labels == 0] = -100
    #labels[labels == -100] = 0

    decoder_attention_mask = batch['labels_attention_mask'].to(device)
    outputs = model_generator.generator(input_ids=preamble_tokens,
                                        attention_mask=preamble_attention_mask,
                                        labels=labels,
                                        decoder_attention_mask=decoder_attention_mask,

                                        )
    # Total loss is the info_loss and the LM loss
    loss = computeLoss(outputs[1], labels, rank_alpha=0.7, mle_only=mle_only,
                       ignore_index=1, padding_idx=1) / accumulation_steps  # .item()
    # print(loss,info_loss)
    # last_hidden_states = outputs.hidden_states[-1]
    # print(last_hidden_states.shape)

    batch_loss = loss.item()  # + info_loss

    loss.backward()

    if (step+1) % accumulation_steps == 0:             # Wait for several backward steps
        # Now we can do an optimizer step
        model_generator.optimizer.step()
        model_generator.scheduler.step()
        model_generator.generator.zero_grad()

    # info_loss.backward()
    # optimizer2.step()

    return batch_loss
# Training step for the fusion model

# Training the MPU based models
def FusionModelsTraining(step, batch):
    met, rate, val = batch['metrics_seq'].to(
        device), batch['rates'].to(device), batch['values'].to(device)
    clb, di = batch['class_labels'].to(
        device), batch['data_info'].to(device)
    met_att = batch['metrics_attention'].to(device)
    rate_att = batch['rate_attention'].to(device)
    val_att = batch['value_attention'].to(device)

    preamble_tokens = batch['preamble_tokens'].to(device)
    preamble_attention_mask = batch['preamble_attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Encode the semantic representation of the table containing the metrics
    table_rep = model_generator.performAuxEncoding([met.detach().clone(), met_att.detach().clone()],
                                                   [val.detach().clone(),
                                                    val_att.detach().clone()],
                                                   [rate.detach().clone(), rate_att.detach().clone()])

    decoder_attention_mask = batch['labels_attention_mask'].to(device)

    outputs = model_generator.generator(input_ids=preamble_tokens,
                                        attention_mask=preamble_attention_mask,
                                        table_inputs=table_rep,
                                        table_attention_mask=None,
                                        labels=labels,
                                        decoder_attention_mask=decoder_attention_mask
                                        )
    # Total loss is the info_loss and the LM loss
    #loss = outputs[0].mean()
    loss = computeLoss(outputs[1], labels, rank_alpha=0.7, mle_only=mle_only,
                       ignore_index=1, padding_idx=1) / accumulation_steps  
    batch_loss = loss.item()  
    loss.backward()
    if (step+1) % accumulation_steps == 0:             # Wait for several backward steps
        # Now we can do an optimizer step
        model_generator.optimizer.step()
        model_generator.scheduler.step()
        model_generator.generator.zero_grad()
        model_generator.aux_encoder.zero_grad()

    return batch_loss
# Set up the function for training the model


def trainNarrator(train_dataset_loader, epochs):
    print('======== Beginning Model Training ======')

    # initialize the time keeper
    total_t0 = time.time()

    training_stats = []
    gama = 0
    for epoch_i in tqdm(range(0, epochs)):
        # ========================================
        #               Training
        # ========================================

        print("")
        print(
            '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model_generator.generator.train()
        model_generator.generator.zero_grad()

        if model_generator.aux_encoder is not None:
            model_generator.aux_encoder.train()
            model_generator.aux_encoder.zero_grad()
        for step, batch in enumerate(train_dataset_loader):
            if model_generator.model_type == 'baseline':
                batch_loss = baselineTraining(step, batch)

            else:
                batch_loss = FusionModelsTraining(step, batch)
            total_train_loss += batch_loss

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataset_loader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-total_t0)))


def generateAndEvaluateYY(seed=43):
    model_generator.generator.eval()
    if model_generator.aux_encoder is not None:
        model_generator.aux_encoder.eval()

    generated_output_dict = {}

    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        generated_outputs = []
        seed_everything(seed)
        for prompt in test_sample:
            tt = prompt
            batch = dataset.processTableInfo(tt)
            # batch=dataset.processTableInfo(test_sample[tidx])
            clb, di = batch['class_labels'].unsqueeze(0).to(
                device), batch['data_info'].unsqueeze(0).to(device)
            met, rate, val = batch['metrics_seq'].unsqueeze(0).to(device), batch['rates'].unsqueeze(
                0).to(device), batch['values'].unsqueeze(0).to(device)
            preamble_tokens = batch['preamble_tokens'].unsqueeze(
                0).to(device)
            preamble_attention_mask = batch['preamble_attention_mask'].unsqueeze(
                0).to(device)
            met_att = batch['metrics_attention'].unsqueeze(0).to(device)
            rate_att = batch['rate_attention'].unsqueeze(0).to(device)
            val_att = batch['value_attention'].unsqueeze(0).to(device)

            if model_generator.aux_encoder is not None:
                table_rep = model_generator.performAuxEncoding(
                    [met, met_att], [val, val_att], [rate, rate_att])
                #table_rep= torch.zeros_like(table_rep)
                sample_outputs = model_generator.generator.generate(input_ids=preamble_tokens,
                                                                    attention_mask=preamble_attention_mask,
                                                                    table_inputs=table_rep,
                                                                    table_attention_mask=None,
                                                                    num_beams=bs,
                                                                    repetition_penalty=1.5,
                                                                    length_penalty=8.6,
                                                                    early_stopping=True,
                                                                    use_cache=True,
                                                                    max_length=190,
                                                                    no_repeat_ngram_size=2,
                                                                    num_return_sequences=1,
                                                                    do_sample=False
                                                                    # bos_token_id=random.randint(1,30000),
                                                                    )
            else:
                sample_outputs = model_generator.generator.generate(input_ids=preamble_tokens,
                                                                    attention_mask=preamble_attention_mask,
                                                                    num_beams=bs,
                                                                    repetition_penalty=1.5,
                                                                    length_penalty=8.6,
                                                                    early_stopping=True,
                                                                    use_cache=True,
                                                                    max_length=190,
                                                                    no_repeat_ngram_size=2,
                                                                    num_return_sequences=1,
                                                                    do_sample=False
                                                                    # bos_token_id=random.randint(1,30000),
                                                                    )
            ss = tokenizer.decode(sample_outputs[0],
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
            generated_outputs.append(ss)

        print(f'Generation based on beam size {bs} is complete')
        #computeParentScore(refs=refs, predicted=generated_outputs, tables=eval_tables)
        print('\n')
        generated_output_dict[bs] = generated_outputs
    return generated_output_dict











# Save the configuration_arguments
json.dump(params_dict, open(output_path+'/arguments.dict', 'w'))
results = {}
if not args.only_eval:
    trainNarrator(train_dataloader, epochs)
    # Save the trained model
    model_generator.saveModel(model_path=output_path+'trained_model.pt')

    # Perform evaluations
    results = generateAndEvaluate(model_generator,test_dataloader,seed=args.seed,sample_too=args.sample_bs)

else:
    print('-- Evaluating Performance --')
    print('-- Please make sure to restore model checkpoint before this step')
    print(output_path+'trained_model.pt')
    if os.path.exists(output_path+'trained_model.pt'):
        model_generator.loadModel(model_path=output_path+'trained_model.pt')
        results = generateAndEvaluate(model_generator,test_dataloader,seed=args.seed,sample_too=args.sample_bs)
        #results = generateAndEvaluate(seed=args.seed)
    else:
        print('-- Model files not found')
if len(results) > 0:
    json.dump(results, open(output_path+'/narrations_outputoe.json', 'w'))
print('File run complete')
