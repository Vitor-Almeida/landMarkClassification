def printLog(idx,cur_epoch,model,accuracy):

    log_interval = int(model.total_steps/50)

    if idx == -1:
        print('| end of epoch {:3d} '
        '| test accuracy {:8.3f}'.format(cur_epoch+1, accuracy))

    if idx % log_interval == 0 and idx > 0:
        print('| epoch {:3d} | {:5d}/{:5d} batches '
                '| train accuracy {:8.3f}'.format(cur_epoch+1, idx, len(model.train_dataloader),
                                            accuracy))

    return None

def printParamsTerminal(model):

    print('-'*22,'Parameters:','-'*22)
    print(f'modelo: {model.model_name} | batchsize: {model.batchsize} | max_tokens = {model.max_char_length} | learning_rate = {model.lr}')
    print(f'epochs = {model.epochs} | warmup_size = {model.warmup_size} | dropout = {model.dropout}')
    print(f'num_labels = {model.num_labels} | dataset_length = {len(model.dataset_train.labels)} | dataset_name = {model.dataset_train.name}')
    print(f'random_seed = {model.seed_val}')
    print('-'*59)

    return None