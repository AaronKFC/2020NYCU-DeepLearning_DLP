from __future__ import unicode_literals, print_function, division
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import json
import string

torch.cuda.empty_cache()
torch.manual_seed(1)
#####################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------Hyper Parameters----------#
vocab_size = 29  #26個字母、sos、eos、unknown token
SOS_token = 0
EOS_token = 27
UNK_token = 28

"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights
========================================================================================"""

def getData(mode):
    if mode == 'train':
        with open('train.json', 'r', encoding='utf-8') as f:
            train_wds = json.load(f)
        inputs = []
        targets = []
        inputs_len = []
        targets_len = []
        for i in range(len(train_wds)):
            for w in train_wds[i]['input']:
                inputs.append(w)
                inputs_len.append(len(w))
                targets.append(train_wds[i]['target'])
                targets_len.append(len(train_wds[i]['target']))
        return np.array(inputs), np.array(targets), np.array(inputs_len), np.array(targets_len)
    else:
        with open('test.json', 'r', encoding='utf-8') as f:
            test_wds = json.load(f)
        inputs = []
        targets = []
        inputs_len = []
        targets_len = []
        for i in range(len(test_wds)):
            for w in test_wds[i]['input']:
                inputs.append(w)
                inputs_len.append(len(w))
                targets.append(test_wds[i]['target'])
                targets_len.append(len(test_wds[i]['target']))
        return np.array(inputs), np.array(targets), np.array(inputs_len), np.array(targets_len)

x_train, y_train, x_train_len, y_train_len = getData('train')
x_test, y_test, x_test_len, y_test_len = getData('test')
word_len = np.hstack([x_train_len, y_train_len, x_test_len, y_test_len])
max_seq_len = int(np.max(word_len) +6) 

#characters = string.digits + string.ascii_uppercase
characters = ' '+string.ascii_lowercase

def letter2index(letter):
    return characters.find(letter)

def word2seqToken(line, eos=True):
    ary = np.zeros(len(line))
    tensor = torch.LongTensor(ary)
    eos_tensor = torch.LongTensor([EOS_token])
    for li, letter in enumerate(line):
        tensor[li] = letter2index(letter)
    if eos:
        return torch.cat((tensor,eos_tensor))
    else:
        return tensor

def formPair(x,y):
    wdlist = []
    tglist = []
    for i in range(len(x)):
        w = word2seqToken(x[i])
        t = word2seqToken(y[i])
        wdlist.append(w)
        tglist.append(t)
    return list(zip(wdlist,tglist))

# training_pairs = formPair(x_train,y_train)
# random.shuffle(training_pairs)

# def letter2onehot(target_idx):
#     ary = np.zeros((1, vocab_size))
#     # tensor = torch.zeros(len(line))
#     tensor = torch.LongTensor(ary)
#     # tensor = torch.zeros(1, vocab_size)
#     # tensor = torch.LongTensor(tensor)
#     tensor[0][target_idx] = 1
#     return tensor
# a = letter2onehot(2)

#for i, (x,y) in enumerate(training_pairs):#,start=1):
#    if i<5:
#        print(x,y)

#b = torch.stack(wdlist) #failed
# print(word2seqToken(x_train[0]))


#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

# def compute_bleu(output, reference):
#     cc = SmoothingFunction()
#     return sentence_bleu([reference], output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1)
# # BLEU_score = compute_bleu(output, reference)


##### Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)#, num_layers=self.n_layers)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1)#(input_size,hidden_size,num_layers)
        # self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=n_layers, bidirectional=True)

    # def forward(self, input, hidden):
    def forward(self, input, hn, cn):
        embedded = self.embedding(input)
        output = embedded.view(1, 1, -1)
        # output, hidden = self.gru(output, hidden)
        output,(hn,cn) = self.lstm(output, (hn,cn))        
        return output, (hn,cn)
        # return output, hidden#原gru

    def initHidden(self):
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return torch.zeros(1, 1, self.hidden_size, device=device)
    # def initHidden(self):
    #     # 开始时刻, 没有隐状态
    #     return (torch.zeros(1, 1, self.hidden_size, device=device),
    #             torch.zeros(1, 1, self.hidden_size, device=device))
    #     # h_n, h_c shape均為(n_layers, batch, hidden_size)   
    #     # LSTM 有两个 hidden states, h_n 是分线, h_c 是主线劇情

##### Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hn, cn):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        # output, hidden = self.gru(output, hidden)
        output,(hn,cn) = self.lstm(output, (hn,cn)) 
        output = self.out(output[0])
        return output, (hn,cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

'''
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_seq_len=max_seq_len):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_seq_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hn,cn, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hn[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        # output, hidden = self.gru(output, hidden)
        output,(hn,cn) = self.lstm(output, (hn,cn))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, (hn,cn), attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
'''

# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_seq_len=max_seq_len):
def train(input_tensor, target_tensor, encoder, decoder, loss_optimizer, criterion, max_seq_len, Tforce=0.8):
    encoder.train()
    decoder.train()
    encoder_hn = encoder.initHidden()
    encoder_cn = encoder.initHidden()

    loss_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # encoder_outputs = torch.zeros(max_seq_len, encoder.hidden_size, device=device)

    loss = 0
    #----------sequence to sequence part for encoder----------#
    # encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    for ei in range(input_length):
        # encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        _, (encoder_hn, encoder_cn) = encoder(input_tensor[ei], encoder_hn, encoder_cn)
        # encoder_outputs[ei] = encoder_output[0][0]  #只能[0][0]

    #----------sequence to sequence part for decoder----------#
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # decoder_hidden = encoder_hidden
    # decoder_hn = decoder.initHidden()
    decoder_hn = encoder_hn
    decoder_cn = encoder_cn
    #----------Teacher forcing part----------#
    use_teacher_forcing = True if random.random() < Tforce else False
	
    if use_teacher_forcing:
        '''##### Teacher forcing: Feed the target as the next input #####'''
        for di in range(target_length):
            # decoder_output, (decoder_hn,decoder_cn), decoder_attention = decoder(
            #                   decoder_input, decoder_hn,decoder_cn, encoder_outputs)  #encoder_outputs要加進來嗎？
            decoder_output, (decoder_hn,decoder_cn)= decoder(decoder_input, decoder_hn,decoder_cn)  #encoder_outputs要加進來嗎？
            
            target_tr = target_tensor[di].unsqueeze(0)  #不unsqueeze的話，dim=None
            loss += criterion(decoder_output, target_tr)
            # loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        '''##### Without teacher forcing: use its own predictions as the next input #####'''
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #                 decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, (decoder_hn,decoder_cn) = decoder(decoder_input, decoder_hn,decoder_cn)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            target_tr = target_tensor[di].unsqueeze(0)  #不unsqueeze的話，dim=None
            loss += criterion(decoder_output, target_tr) 
            # loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    # encoder_optimizer.step()
    # decoder_optimizer.step()
    loss_optimizer.step()
    return loss.item() / target_length


def evaluate(test_pairs, encoder, decoder, max_seq_len=max_seq_len, output_words=False):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        decoder_words = []
        test_BLEU = []
        for ti in range(len(y_test)):
            test_pair = test_pairs[ti]
            input_tensor = test_pair[0].to(device)
            target_word = y_test[ti]
            
            #----------sequence to sequence part for encoder----------#
            encoder_hn = encoder.initHidden()
            encoder_cn = encoder.initHidden()
            input_length = input_tensor.size(0)
            for ei in range(input_length):
                _, (encoder_hn, encoder_cn) = encoder(input_tensor[ei], encoder_hn, encoder_cn)
            
            #----------sequence to sequence part for decoder----------#
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hn = encoder_hn
            decoder_cn = encoder_cn
            decoded_letters = []
            for di in range(max_seq_len):
                decoder_output, (decoder_hn,decoder_cn)= decoder(
                                decoder_input, decoder_hn,decoder_cn)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    break
                else:
                    decoded_letters.append(characters[topi.item()])
                decoder_input = topi.squeeze().detach()
            ##### token to word #####
            decoder_word = ''.join(decoded_letters)
            decoder_words.append(decoder_word)
            ##### BLEU-4 Score Calculation #####
            word_BLEU = compute_bleu(decoder_word, target_word)
            test_BLEU.append(word_BLEU)
        avg_test_BLEU = np.average(test_BLEU)
        if output_words:
            return decoder_words, avg_test_BLEU
        else:
            return avg_test_BLEU


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=200, learning_rate=0.001, model_name='seq2seq', epoch=0, Tforce=0.8, plot_test_BLEU=[]):
    start = time.time()
    plot_losses = []
    # plot_test_BLEU = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    loss_optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    # loss_optimizer = optim.Adagrad(list(encoder.parameters()) + list(decoder.parameters()),
    #                                 lr=learning_rate, lr_decay=0, weight_decay=0)
    
    ########### your own dataloader ##########
    training_pairs = formPair(x_train,y_train)
    test_pairs = formPair(x_test,y_test)

    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        random.shuffle(training_pairs)
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0].to(device)
        # print(input_tensor)
        target_tensor = training_pair[1].to(device)
        
        ####### Training losses Computation ########
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, loss_optimizer, criterion, max_seq_len, Tforce)
                     # decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            ##### Evaluation #####
            test_BLEU = evaluate(test_pairs, encoder, decoder, max_seq_len=max_seq_len)
            print(f'BLEU-4 score = {test_BLEU:.4f}')
            
        
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            ##### Evaluation #####
            test_BLEU_tp = evaluate(test_pairs, encoder, decoder, max_seq_len=max_seq_len)
            test_BLEU_tp = np.round(test_BLEU_tp,3)
            plot_test_BLEU.append(test_BLEU_tp)
            
            ##### Save Model #####
            if test_BLEU_tp >0.82 and test_BLEU_tp==max(plot_test_BLEU):
                model_path = 'save_model/weights_'
                model_fn = model_name +'_hid' +str(hidden_size) +'_Tforce' +str(Tforce) +'_epo' +str(epoch) +'_bleu' +str(test_BLEU_tp) +'.h5'
                torch.save(encoder.state_dict(),(model_path +'encoder_' +model_fn))
                torch.save(decoder.state_dict(),(model_path +'decoder_' +model_fn))
                torch.save(loss_optimizer.state_dict(),(model_path +'optim_' +model_fn))
                print("Save the best model weights at [Tforce={}, Epoch={}, Iter={}]".format(Tforce, epoch, iter))
        
    return plot_losses, plot_test_BLEU


################# Start Training Process ####################
model_name = 'S2S_LSTM'
hidden_size = 256
epochs = 30
print_every=1000
plot_every=1000
num_iters = len(x_train)  #12925
Tforce = np.linspace(0.86, 0.6, epochs+1)  #teacher_forcing_ratio

encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, vocab_size, dropout_p=0.1).to(device)

start_epo = 1
plot_train_loss = []
plot_test_BLEU = []
for ep in range(start_epo, epochs+1):    
    if ep <= 12:
        lr = 0.05
    elif ep <= 20:
        lr = 0.03
    elif ep <= 26:
        lr = 0.01
    else:
        lr = 0.005
    tf_ratio = np.round(Tforce[ep],3)
    print(f'\nEpoch={ep:2d}, Learn_rate={lr}, Tforce={tf_ratio:.4f}')
    
    plot_loss, plot_BLEU = trainIters(encoder1, decoder1, n_iters=num_iters, 
                                print_every=print_every, plot_every=plot_every, 
                                learning_rate=lr, model_name=model_name, epoch=ep, Tforce=tf_ratio,
                                plot_test_BLEU=plot_test_BLEU)
    plot_train_loss.extend(plot_loss)
    plot_test_BLEU = []
    plot_test_BLEU.extend(plot_BLEU)
    
    test_pairs=formPair(x_test,y_test)
    decoder_words, avg_test_BLEU = evaluate(test_pairs, encoder1, decoder1, 
                                            max_seq_len=max_seq_len, output_words=True)
    print(f'avg_test_BLEU:{avg_test_BLEU:.3f}')

############ save acc history ###############
# Tforce = teacher_forcing_ratio
path = 'results/'
train_loss_ary = np.array(plot_train_loss)
train_loss_his = pd.DataFrame(train_loss_ary)
train_loss_fn = path + model_name +'_trainLoss' +'_hid' +str(hidden_size) +'_Tforce' +str(Tforce[0])   +'_part1'
# train_loss_fn = path + model_name +'_trainLoss' +'_TforceDecay' +'_part2'
train_loss_his.to_csv(train_loss_fn +'.csv', index=False)

test_BLEU_ary = np.array(plot_test_BLEU)
test_BLEU_his = pd.DataFrame(test_BLEU_ary)
test_BLEU_fn = path +model_name +'_testBLEU' +'_hid' +str(hidden_size) +'_Tforce' +str(Tforce[0])     +'_part1'
# test_BLEU_fn = path +model_name +'_testBLEU' +'_TforceDecay'   +'_part2'
test_BLEU_his.to_csv(test_BLEU_fn +'.csv', index=False)

##################################################
####### Training History Plot ############
def loss_history(train_loss, model_name):
    # epo = np.linspace(1, epoch, epoch)
    plt.plot(train_loss)
    # plt.plot(epo, test_acc)
    plt.title('Training Loss History--' +model_name)  
    plt.ylabel('Loss')
    plt.xlabel('Iterations (per1000)')
    # plt.legend(['train_acc', 'test_acc'], loc='lower right')#, fontsize='small')
    # plt.grid()
    plt.show()

def BLEU_history(test_BLEU, model_name):
    plt.plot(test_BLEU)
    plt.title('Test Average BLEU-4 Score History--' +model_name)  
    plt.ylabel('BLEU-4 Score')
    plt.xlabel('Iterations (per1000)')
    plt.show()

loss_history(train_loss_ary, model_name)
BLEU_history(test_BLEU_ary, model_name)

'''
######## Inference Testing #########
encoder_infer = EncoderRNN(vocab_size, hidden_size).to(device)
decoder_infer = DecoderRNN(hidden_size, vocab_size).to(device)
# attn_decoder_infer = AttnDecoderRNN(hidden_size, vocab_size, dropout_p=0.1).to(device)

encoder_infer.load_state_dict(torch.load(
                  'save_model/weights_encoder_S2S_LSTM_hid256_Tforce0.6_epo30_bleu0.973.h5'))
decoder_infer.load_state_dict(torch.load(
                  'save_model/weights_decoder_S2S_LSTM_hid256_Tforce0.6_epo30_bleu0.973.h5'))

test_pairs=formPair(x_test,y_test)
decoder_words_infer, avg_test_BLEU_infer = evaluate(test_pairs, encoder_infer, decoder_infer, 
                                            max_seq_len=max_seq_len, output_words=True)
decoder_words_infer = np.array(decoder_words_infer)
word_compa = np.vstack([x_test, y_test, decoder_words_infer])
words_compa = pd.DataFrame(word_compa.T, columns=['input', 'target', 'pred'])
word_path = 'result_words/'
words_compa_fn = word_path + model_name +'_Tforce0.75_epo39_bleu0.865'
words_compa.to_csv(words_compa_fn +'.csv', index=False)
print('\nBLEU-4 score of inference phase')
print(avg_test_BLEU_infer)

avg_test_BLEU_infer = np.round(avg_test_BLEU_infer,4)
print('\nBLEU-4 score of inference phase')
print(avg_test_BLEU_infer)

for i in range(len(x_test)):
    print('='*26)
    print(f'input:  {x_test[i]}')
    print(f'target: {y_test[i]}')
    print(f'pred:   {decoder_words_infer[i]}')
print('='*26)
print(f'BLEU-4 score:{avg_test_BLEU_infer}')
'''

# '''
# ### Part Separable Training History Combine
# loss_part1 = pd.read_csv((path+'S2S_LSTM_Attn_trainLoss_Tforce0.75_part1.csv'))#,header=None)
# loss_part1 = loss_part1.values
# bleu_part1 = pd.read_csv((path+'S2S_LSTM_Attn_testBLEU_Tforce0.75_part1.csv'))
# bleu_part1 = bleu_part1.values
# loss_part2 = pd.read_csv((path+'S2S_LSTM_Attn_trainLoss_Tforce0.75_part2.csv'))
# loss_part2 = loss_part2.values
# bleu_part2 = pd.read_csv((path+'S2S_LSTM_Attn_testBLEU_Tforce0.75_part2.csv'))
# bleu_part2 = bleu_part2.values
# loss_part2 = pd.read_csv((path+'S2S_LSTM_Attn_trainLoss_Tforce0.75_part2.csv'))
# loss_part2 = loss_part2.values
# bleu_part2 = pd.read_csv((path+'S2S_LSTM_Attn_testBLEU_Tforce0.75_part2.csv'))
# bleu_part2 = bleu_part2.values

# loss_all = np.vstack([loss_part1,loss_part2])
# bleu_all = np.vstack([bleu_part1,bleu_part2])

# loss_history(loss_all, model_name)
# BLEU_history(bleu_all, model_name)

# '''

###################################################################################
######## save the best model weights
##### sample code for save all models 
# savefilename = model_name +str(epoch)+'.tar'
# torch.save({'epoch':epoch, 'state_dict':model.state_dict() }, savefilename)
# ##### code for save the best model
# # model_fn = 'save_model/bestweights_' +model_name +'_batch' +str(args.batch_size) +'_totEpo' +str(args.epochs) +'bestEpo' +str(best_epoch) +'.h5'
# model_path = 'save_model/weights_'
# model_fn = model_name +'_Tforce' +str(Tforce) +'.h5'
# torch.save(encoder.state_dict(),(model_path +'encoder_' +model_fn))
# torch.save(decoder.state_dict(),(model_path +'decoder_' +model_fn))
# torch.save(loss_optimizer.state_dict(),(model_path +'optim_' +model_fn))
# print("Save the model at iter {}".format(iteration))
