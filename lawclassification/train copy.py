import transformers
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset.datasetLoad import yelpReview
from models.models import normalBert
from utils.helperfuns import get_time, flat_accuracy, format_time, print_model_layers
from models.train_text import train_text

def main():

    model_name = "bert-base-uncased"
    batchsize=16
    max_char_length = 128
    lr = 2e-5
    epochs = 5
    warmup_size = 0.1

    train_text(model_name, batchsize, max_char_length, lr, epochs, warmup_size, dataset, model)

    #################################ARRUMAR ESSA BOSTA: ##########################################
    fix_this = yelpReview(typeSplit='val',max_length = max_length, tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True) )
    fix_this_dataloader = DataLoader(dataset=fix_this,batch_size=batchsize)
    ###############################################################################################



    num_labels = len(np.unique(fix_this.target))
    
    
    total_steps = len(fix_this_dataloader) * epochs
    
    NormalModel=normalBert(num_labels,lr,epochs,total_steps)
    tokenizer = NormalModel.tokenizer
    device = NormalModel.device
    optimizer = NormalModel.optimizer
    scheduler = NormalModel.scheduler
    model = NormalModel.bert_model

    training_stats = []
    total_t0 = get_time()

    data_val = yelpReview(typeSplit='val',max_length = max_length, tokenizer = tokenizer)
    data_test = yelpReview(typeSplit='test',max_length = max_length, tokenizer = tokenizer)
    data_train = yelpReview(typeSplit='train',max_length = max_length, tokenizer = tokenizer)

    val_dataloader = DataLoader(dataset=data_val,batch_size=batchsize)
    test_dataloader = DataLoader(dataset=data_test,batch_size=batchsize)
    train_dataloader = DataLoader(dataset=data_train,batch_size=batchsize,shuffle=True)


    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = get_time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(get_time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 

            b_input_ids = batch['ids'].to(device)
            b_input_ids = torch.squeeze(b_input_ids,1)
            b_input_mask = batch['mask'].to(device)
            b_input_mask = torch.squeeze(b_input_mask,1)
            b_labels = batch['target'].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            part_result = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            loss = part_result['loss']
            logits = part_result['logits']

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(get_time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = get_time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in val_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch['ids'].to(device)
            b_input_ids = torch.squeeze(b_input_ids,1)
            b_input_mask = batch['mask'].to(device)
            b_input_mask = torch.squeeze(b_input_mask,1)
            b_labels = batch['target'].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                part_result_val = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                loss = part_result_val['loss']
                logits = part_result_val['logits']
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(get_time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(get_time()-total_t0)))



    return None


if __name__ == '__main__':
    main()