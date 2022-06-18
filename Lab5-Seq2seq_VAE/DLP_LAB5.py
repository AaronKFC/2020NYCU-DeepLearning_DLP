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
import string

torch.cuda.empty_cache()
# torch.manual_seed(1)
#####################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------Hyper Parameters----------#
vocab_size = 28  #26個字母、sos、eos、unknown token
SOS_token = 0
EOS_token = 27

"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""

def getData(mode):
    if mode == 'train':
        train_wds = pd.read_table('dataset/train.txt',sep=' ',header=None) #读入txt文件，分隔符为' '(空格)
        train_wds = train_wds.values
        inputs = []
        tense = []
        inputs_len = []
        for i in range(train_wds.shape[0]):
            for j in range(train_wds.shape[1]):
                w = train_wds[i,j]
                inputs.append(w)
                inputs_len.append(len(w))
                tense.append(j)
        return np.array(inputs), np.array(tense), np.array(inputs_len)
    else:
        test_wds = pd.read_table('dataset/test.txt',sep=' ',header=None)
        test_wds = test_wds.values
        
        inputs = test_wds[:,0]
        inputs_tense = test_wds[:,2]
        targets = test_wds[:,1]
        targets_tense = test_wds[:,3]
        return inputs, inputs_tense, targets, targets_tense

x_train, x_tense, x_train_len = getData('train')
x_test, x_test_tense, y_test, y_test_tense = getData('test')
# max_seq_len = np.max(x_train_len) +5
max_seq_len = 15+5

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

def formPair(x_train, x_tense, y_train, y_tense):
    w_inp_list = []
    t_inp_list = []
    w_tar_list = []
    t_tar_list = []
    for i in range(len(x_train)):
        w_inp = word2seqToken(x_train[i])
        t_inp = x_tense[i]
        w_inp_list.append(w_inp)
        t_inp_list.append(t_inp)
        
        w_tar = word2seqToken(x_train[i])
        t_tar = y_tense[i]
        w_tar_list.append(w_tar)
        t_tar_list.append(t_tar)
    return list(zip(zip(w_inp_list,t_inp_list),zip(w_tar_list,t_tar_list)))

training_pairs = formPair(x_train, x_tense, x_train, x_tense)
test_pairs = formPair(x_test, x_test_tense, y_test, y_test_tense)
'''
training_pairs[i][j][k]
i-dim=0,1: index for [(input,input_tense),(target,target_tense)] pair
j-dim=0,1: index for (input,target) pair
k-dim=0,1: index for (word_tensor, tense) pair
'''

# N_TENSE=4
# def idx2onehot(idx, n=N_TENSE):
#     assert idx.shape[1] == 1
#     assert torch.max(idx).item() < n
#     onehot = torch.zeros(idx.size(0), n)
#     onehot.scatter_(1, idx.data, 1)
#     return onehot

#N_TENSE=4
#def tense2onehot(idx, n=N_TENSE):
#    # ary = np.zeros((1,1, n))
#    # ary[0][0][target_idx] = 1
#    # tensor = torch.LongTensor(ary)
#    tensor = torch.zeros(1, 1, n)
#    tensor[0][0][idx] = 1
#    return tensor  #''' 是否需要squeeze? '''

##### compute BLEU-4 score #####
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""
def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'dataset/train.txt' #should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            # print(t)
            for i in words_list:
                # print(i)
                if t == i:
                    score += 1
    return score/len(words)


##### Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_dim, n_tense_embed):
        super(EncoderRNN, self).__init__()
        # self.Cond_Embed = Cond_Embed
        self.hidden_size = hidden_size
        self.input_size = hidden_size #為word embedding後的dim
        # self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.n_tense_embed = n_tense_embed
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM((self.input_size+n_tense_embed), (hidden_size +n_tense_embed), num_layers=1)  #(input_size,hidden_size,num_layers)

    def forward(self, input1, input2, hn, cn):
        c_embed = input1.view(1, 1, -1)
        embedded = self.embedding(input2)
        wd_embed = embedded.view(1, 1, -1)  # [batch, seq, embedding_size]
        c_wd_inp = torch.cat((c_embed, wd_embed), dim=2)
        
        output,(hn,cn) = self.lstm(c_wd_inp, (hn,cn))
        return output, (hn,cn)

    def initHidden(self):
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return torch.zeros(1, 1, self.hidden_size, device=device) #''' 未concate condition '''


##### Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, latent_dim, n_tense_embed):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size #為word embedding後的dim
        self.latent_dim = latent_dim
        self.n_tense_embed = n_tense_embed
        
        self.latent_to_hidden = nn.Linear((latent_dim +n_tense_embed), (hidden_size +n_tense_embed))
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM((self.input_size+latent_dim +n_tense_embed), (hidden_size +n_tense_embed), num_layers=1)
        self.out = nn.Linear((hidden_size +n_tense_embed), vocab_size)

    def forward(self, input1, input2, hn, cn):
        latent_z = input1.view(1, 1, -1)
        wd_embed = self.embedding(input2).view(1, 1, -1)
        wd_embed = F.relu(wd_embed)
        lz_wd_inp = torch.cat((latent_z, wd_embed), dim=2)
        
        output,(hn,cn) = self.lstm(lz_wd_inp, (hn,cn)) 
        output = self.out(output[0])
        return output, (hn,cn)

    def initHidden(self):
        return torch.zeros(1, 1, (self.n_tense_embed +self.hidden_size), device=device) #''' 未concate condition '''


class Reparame(nn.Module):
    def __init__(self, COND_EMB_DIM, hidden_size, latent_length):
        super(Reparame, self).__init__()
        self.COND_EMB_DIM = COND_EMB_DIM
        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.COND_EMB_DIM +self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.COND_EMB_DIM +self.hidden_size, self.latent_length)

    def forward(self, cell_output):
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        std = torch.exp(self.latent_logvar / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(self.latent_mean)
        return self.latent_mean, self.latent_logvar, x_sample


class Cond_Embed(nn.Module):
    def __init__(self, N_TENSE, COND_EMB_DIM, pad_id=0, dropout=0.1):
        super(Cond_Embed, self).__init__()
        self.embedding = nn.Embedding(N_TENSE, COND_EMB_DIM, padding_idx=pad_id)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, input):  # [batch, seq]
        tense_embed = self.embedding(input).view(1, 1, -1)
        tense_embed = self.dropout(tense_embed)
        return tense_embed  # [batch, seq, embedding_size]


########### Loss Function #############
def KLD_loss(mean, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD


def train_CVAE(input_wdts, target_wdts, encoder, decoder, Cond_Embed, Reparame, criterion, loss_optimizer, max_seq_len, Tforce=0.8, kl_anl=0.5):
    encoder.train()
    decoder.train()
    Reparame.train()
    Cond_Embed.train()

    loss_optimizer.zero_grad()
    wd_inp = input_wdts[0].to(device)
    ts_inp = torch.LongTensor([input_wdts[1]]).to(device)
    wd_tar = target_wdts[0].to(device)
    ts_tar = torch.LongTensor([target_wdts[1]]).to(device)
        
    wd_inp_len = wd_inp.size(0)
    wd_tar_len = wd_tar.size(0)
    
    # c_inp = tense2onehot(ts_inp).to(device)
    c_inp = Cond_Embed(ts_inp)
    encoder_hn = torch.cat((c_inp, encoder.initHidden()), dim=2)
    encoder_cn = torch.zeros(1, 1, (256 + 8), requires_grad=True, device=device)
    
    recons_loss = 0
    #----------sequence to sequence part for encoder----------#
    for ei in range(wd_inp_len):
        _, (encoder_hn, encoder_cn) = encoder(c_inp, wd_inp[ei], encoder_hn, encoder_cn)
        # _, (encoder_hn, encoder_cn), z_mu, z_var = encoder(wd_inp[ei], encoder_hn, encoder_cn)
        
    #----------sequence to sequence part for decoder----------#    
    # sample from the distribution having latent parameters z_mu, z_var
    ####### reparameterize ########
    z_mu, z_var, x_sample = Reparame(encoder_hn)
    x_sample.unsqueeze(0).unsqueeze(0)
    
    c_tar = Cond_Embed(ts_tar)
    latent_z = torch.cat((c_tar, x_sample), dim=2)
    
    decoder_hn = decoder.latent_to_hidden(latent_z)
    decoder_cn = torch.zeros(1, 1, (256 +8), requires_grad=True).to(device)
    
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    #----------Teacher forcing part----------#
    use_teacher_forcing = True if random.random() < Tforce else False
	
    if use_teacher_forcing:
        '''##### Teacher forcing: Feed the target as the next input #####'''
        for di in range(wd_tar_len):
            # print(di)
            decoder_output, (decoder_hn,decoder_cn)= decoder(latent_z, decoder_input, decoder_hn,decoder_cn)  #encoder_outputs要加進來嗎？
            
            ####### Calculating Loss ########
            wd_target = wd_tar[di].unsqueeze(0)  #不unsqueeze的話，dim=None
            recons_loss += criterion(decoder_output, wd_target)
            decoder_input = wd_tar[di]  # Teacher forcing
    else:
        '''##### Without teacher forcing: use its own predictions as the next input #####'''
        for di in range(wd_tar_len):
            decoder_output, (decoder_hn,decoder_cn) = decoder(latent_z, decoder_input, decoder_hn,decoder_cn)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # print(decoder_output.size())
            ####### Calculating Loss ########
            wd_target = wd_tar[di].unsqueeze(0)  #不unsqueeze的話，dim=None
            # print(wd_target.size())
            recons_loss += criterion(decoder_output, wd_target) 
            if decoder_input.item() == EOS_token:
                break
            
    ###### Loss Calculation #######
    # RC_loss = recons_loss / (di +1)
    RC_loss = recons_loss
    KL_loss = KLD_loss(z_mu, z_var)
    tot_loss = RC_loss + kl_anl*KL_loss
    tot_loss.backward()
    
    loss_optimizer.step()
    return RC_loss.item()/wd_tar_len, KL_loss.item()
    # return RC_loss.item()/wd_tar_len, KL_loss/wd_tar_len


def evaluate(test_pairs, encoder, decoder, Cond_Embed, Reparame, max_seq_len=max_seq_len, output_words=False):
    encoder.eval()
    decoder.eval()
    Reparame.eval()
    Cond_Embed.eval()
    with torch.no_grad():
        decoder_words = []
        test_BLEU = []
        for ti in range(len(y_test)):
            test_pair = test_pairs[ti]
            input_wdts = test_pair[0]
            target_wdts = test_pair[1]
            
            wd_inp = input_wdts[0].to(device)
            ts_inp = torch.LongTensor([input_wdts[1]]).to(device)
            # wd_tar = target_wdts[0].to(device)
            ts_tar = torch.LongTensor([target_wdts[1]]).to(device)
            
            # wd_inp = test_pair[0].to(device)
            wd_tar_letter = y_test[ti] 
            
            #----------sequence to sequence part for encoder----------#
            # c_inp = tense2onehot(ts_inp).to(device)
            c_inp = Cond_Embed(ts_inp)
            encoder_hn = torch.cat((c_inp, encoder.initHidden()), dim=2)
            encoder_cn = torch.zeros(1, 1, (256 + 8), device=device)
            
            wd_inp_len = wd_inp.size(0)
            for ei in range(wd_inp_len):
                _, (encoder_hn, encoder_cn) = encoder(c_inp, wd_inp[ei], encoder_hn, encoder_cn)
                # _, (encoder_hn, encoder_cn), z_mu, z_var = encoder(wd_inp[ei], encoder_hn, encoder_cn)
            
            #----------sequence to sequence part for decoder----------#
            z_mu, z_var, x_sample = Reparame(encoder_hn)
            x_sample.unsqueeze(0).unsqueeze(0)
            
            # c_tar = tense2onehot(ts_tar).to(device)
            c_tar = Cond_Embed(ts_tar)
            latent_z = torch.cat((c_tar, x_sample), dim=2)
            
            decoder_hn = decoder.latent_to_hidden(latent_z)
            decoder_cn = torch.zeros(1, 1, (256 +8)).to(device)  
            
            decoder_input = torch.tensor([[SOS_token]], device=device)
            
            decoded_letters = []
            for di in range(max_seq_len):
                decoder_output, (decoder_hn,decoder_cn)= decoder(latent_z, decoder_input, decoder_hn,decoder_cn)
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
            word_BLEU = compute_bleu(decoder_word, wd_tar_letter)
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


def trainIters(encoder, decoder, Cond_Embed, Reparame, n_iters, print_every=1000, plot_every=200, learning_rate=0.001, 
               model_name='seq2seq', epoch=0, Tforce=0.8, plot_test_BLEU=[], KL_anneal=0.5, latent_dim=64):
    start = time.time()
    plot_losses = []
    plot_RC_losses = []
    plot_KL_losses = []
    # plot_test_BLEU = []
    loss_tot = 0  # Reset every print_every
    plot_loss_tot = 0  # Reset every plot_every
    loss_RC = 0
    plot_loss_RC = 0
    loss_KL = 0
    plot_loss_KL = 0

    loss_optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()) + list(Cond_Embed.parameters()) + list(Reparame.parameters()), lr=learning_rate)
    
    ########### your own dataloader ##########
    training_pairs = formPair(x_train, x_tense, x_train, x_tense)
    test_pairs = formPair(x_test, x_test_tense, y_test, y_test_tense)
    ########### reconstruction loss ##########
    criterion = nn.CrossEntropyLoss()
    
    for iter in range(1, n_iters + 1):
        random.shuffle(training_pairs)
        training_pair = training_pairs[iter - 1]
        input_wdts = training_pair[0]
        target_wdts = training_pair[1]
        
        kl_anl = KL_anneal[(epoch-1)*n_iters+(iter-1)]
        kl_anl = np.round(kl_anl,4)
        
        ####### Training losses Computation ########
        RC_loss, KL_loss = train_CVAE(input_wdts, target_wdts, encoder, decoder, Cond_Embed, Reparame,
                                      criterion, loss_optimizer, max_seq_len, Tforce, kl_anl)
        loss = RC_loss + KL_loss
        loss_tot += loss
        plot_loss_tot += loss
        
        loss_RC += RC_loss
        plot_loss_RC += RC_loss
        loss_KL += KL_loss
        plot_loss_KL += KL_loss

        if iter % print_every == 0 or iter == n_iters:
            loss_tot_avg = loss_tot / print_every
            loss_RC_avg = loss_RC / print_every
            loss_KL_avg = loss_KL / print_every
            loss_tot = 0
            loss_RC = 0
            loss_KL = 0
            print('%s (%d %d%%) [Loss:%.3f, RC:%.3f, KL:%.3f]' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, loss_tot_avg, loss_RC_avg, loss_KL_avg))
            ##### Evaluation #####
            test_BLEU = evaluate(test_pairs, encoder, decoder, Cond_Embed, Reparame, max_seq_len=max_seq_len)
            decoder_words = Gaussian_Gen(decoder, Cond_Embed, max_seq_len=max_seq_len, latent_dim=latent_dim)
            Gaussian_s = Gaussian_score(decoder_words)
            print(f'Anneal={kl_anl:.3f},  BLEU-4 score={test_BLEU:.4f}, G_score={Gaussian_s:.2f}')
            
        # if iter % plot_every == 0 or iter == n_iters:
            plot_loss_tot_avg = plot_loss_tot / plot_every
            plot_losses.append(plot_loss_tot_avg)
            plot_loss_tot = 0
            plot_loss_RC_avg = plot_loss_RC / plot_every
            plot_RC_losses.append(plot_loss_RC_avg)
            plot_loss_RC = 0
            plot_loss_KL_avg = plot_loss_KL / plot_every
            plot_KL_losses.append(plot_loss_KL_avg)
            plot_loss_KL = 0
            
            ##### Evaluation #####
            test_BLEU_tp = evaluate(test_pairs, encoder, decoder, Cond_Embed, Reparame, max_seq_len=max_seq_len)
            test_BLEU_tp = np.round(test_BLEU_tp,3)
            plot_test_BLEU.append(test_BLEU_tp)
            
            ##### Save Model #####
            if test_BLEU_tp >0.5 or Gaussian_s >=0.05: #and test_BLEU_tp <0.75:
                model_path = 'save_model_rc_p4/weights_'
                model_fn = model_name +'_hid' +str(hidden_size) +'_Tforce' +str(Tforce) +'_epo' +str(epoch) +'_bleu' +str(test_BLEU_tp) +'_rc_p4' +'.h5'
                torch.save(encoder.state_dict(),(model_path +'encoder_' +model_fn))
                torch.save(decoder.state_dict(),(model_path +'decoder_' +model_fn))
                torch.save(Reparame.state_dict(),(model_path +'Reparame_' +model_fn))
                torch.save(Cond_Embed.state_dict(),(model_path +'CondEmbed_' +model_fn))
                print("Save the best model weights at [Tforce={}, Epoch={}, Iter={}]".format(Tforce, epoch, iter))
        
    return plot_losses, plot_RC_losses, plot_KL_losses, plot_test_BLEU


###### Gaussian Generator
def Gaussian_Gen(decoder, Cond_Embed, max_seq_len, latent_dim):
    decoder.eval()
    Cond_Embed.eval()
    with torch.no_grad():
        decoder_words = []
        for i in range(100):
            x_sample = torch.randn(1, 1, latent_dim).to(device)
            decoder_cn = torch.zeros(1, 1, (256 +8)).to(device)
            wd4ts = []
            for j in range(4):
                ts_tar = torch.LongTensor([j]).to(device)
                c_tar = Cond_Embed(ts_tar)
                latent_z = torch.cat((c_tar, x_sample), dim=2)
                
                decoder_hn = decoder.latent_to_hidden(latent_z)
                decoder_input = torch.tensor([[SOS_token]], device=device)
                decoded_letters = []
                for di in range(16):
                    decoder_output, (decoder_hn,decoder_cn)= decoder(
                                        latent_z, decoder_input, decoder_hn,decoder_cn)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        break
                    else:
                        decoded_letters.append(characters[topi.item()])
                    decoder_input = topi.squeeze().detach()
                ##### token to word #####
                decoder_word = ''.join(decoded_letters)
                wd4ts.append(decoder_word)
            decoder_words.append(wd4ts)
        return decoder_words


######### KLD_Weight #########
KLD_max_weight = 0.33
def KL_anneal(mode, max_wet = KLD_max_weight):
    if mode == 'mono':
        linear = np.linspace(0, max_wet, 10000)
        horizon = np.linspace(max_wet, max_wet, 390000)
        KL_anneal_mono = np.hstack([linear,horizon])
        return KL_anneal_mono
        
    else:
        linear = np.linspace(0, max_wet, 10000)
        horizon = np.linspace(max_wet, max_wet, 10000)
        KL_anneal = np.hstack([linear, horizon])
        for i in range(20):
            if i==0:
                KL_anneal_cycl = KL_anneal
            else:
                KL_anneal_cycl = np.hstack([KL_anneal_cycl,KL_anneal])
        return KL_anneal_cycl


################# Start Training Process ####################
model_name = 'S2S_CVAE'
hidden_size = 256
N_TENSE = 4
COND_EMB_DIM = 8
LATENT_DIM = 32                              # latent vector dimension
epochs = 21
num_iters = len(x_train)  #4908
Tforce = np.linspace(0.5, 0.5, epochs+1)
KLD_max_weight = 0.33
ANL_mode = 'mono'                          #'mono'
lr = 0.05

KL_anneal = KL_anneal(ANL_mode, KLD_max_weight) 

print_every=1000
plot_every=1000

encoder1 = EncoderRNN(vocab_size, hidden_size, LATENT_DIM, COND_EMB_DIM).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size, LATENT_DIM, COND_EMB_DIM).to(device)
Reparame = Reparame(COND_EMB_DIM, hidden_size, LATENT_DIM).to(device)
Cond_Embed = Cond_Embed(N_TENSE, COND_EMB_DIM, pad_id=0, dropout=0.1).to(device)


start_epo = 1
plot_train_loss_tot = []
plot_train_loss_RC = []
plot_train_loss_KL = []
plot_test_BLEU = []
for ep in range(start_epo, epochs+1):    
    if ep <= 20:
        lr = 0.05
    elif ep <= 40:
        lr = 0.05
    else:
        lr = 0.05
    tf_ratio = np.round(Tforce[ep-1],3)
    print(f'\nEpoch={ep:2d}, Learn_rate={lr}, Tforce={tf_ratio:.4f}')
    
    plot_loss_tot, plot_loss_RC, plot_loss_KL, plot_BLEU = trainIters(encoder1, decoder1, Cond_Embed, Reparame,
                                n_iters=num_iters, print_every=print_every, plot_every=plot_every, 
                                learning_rate=lr, model_name=model_name, epoch=ep, Tforce=tf_ratio,
                                plot_test_BLEU=plot_test_BLEU, KL_anneal=KL_anneal, latent_dim=LATENT_DIM)
    
    plot_train_loss_tot.extend(plot_loss_tot)
    plot_train_loss_RC.extend(plot_loss_RC)
    plot_train_loss_KL.extend(plot_loss_KL)
    plot_test_BLEU = []
    plot_test_BLEU.extend(plot_BLEU)
    
    ##### validation
    test_pairs = formPair(x_test, x_test_tense, y_test, y_test_tense)
    decoder_words, avg_test_BLEU = evaluate(test_pairs, encoder1, decoder1, Cond_Embed, Reparame,
                                            max_seq_len=max_seq_len, output_words=True)
    print(f'avg_test_BLEU:{avg_test_BLEU:.3f}')


