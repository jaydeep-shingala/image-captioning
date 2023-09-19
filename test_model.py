from transformer import Transformer # this is the transformer.py file
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast
import matplotlib.pyplot as plt

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ';',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}

image_features = np.load("Image_feature_tensors.npy")
image_features = [torch.from_numpy(np_array) for np_array in image_features]
image_features = image_features[0:5]
image_features = [tensor.squeeze(0) for tensor in image_features]
print(len(image_features))


with open("Image_text_captions", 'r') as file:
    captions = file.readlines()
captions = [sentence.rstrip('\n').lower() for sentence in captions]
captions = captions[0:5]
print(len(captions))

d_model = 768
batch_size = 50
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.2
num_layers = 1
max_sequence_length = 577
en_vocab_size = len(english_vocabulary)

# transformer = torch.load("image_caption_transformer_model.pt")
transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          en_vocab_size,
                          english_to_index,
                          english_to_index,
                          START_TOKEN, 
                          END_TOKEN, 
                          PADDING_TOKEN) 
transformer.load_state_dict(torch.load("image_caption_transformer_model.pt"))
transformer.eval()
# transformer = Transformer(d_model, 
#                           ffn_hidden,
#                           num_heads, 
#                           drop_prob, 
#                           num_layers, 
#                           max_sequence_length,
#                           en_vocab_size,
#                           english_to_index,
#                           english_to_index,
#                           START_TOKEN, 
#                           END_TOKEN, 
#                           PADDING_TOKEN)    


class TextDataset(Dataset):

    def __init__(self, image_features, captions):
        self.image_features = image_features
        self.captions = captions

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        return self.image_features[idx], self.captions[idx]
    
    
dataset = TextDataset(image_features, captions)
# print(len(dataset))
# print(dataset[1])
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

criterian = nn.CrossEntropyLoss(ignore_index=english_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
# for params in transformer.parameters():
#     if params.dim() > 1:
#         nn.init.xavier_uniform_(params)

# optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NEG_INFTY = -1e9

def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

predicted_captions = []
# actual_captions = []
   
transformer.to(device)   
iterator = iter(train_loader)
epoch_loss = 0

for batch_num, batch in enumerate(iterator):
    eng_batch, kn_batch = batch
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, kn_batch)
    # optim.zero_grad()
    kn_predictions = transformer(eng_batch,
                                    kn_batch,
                                    encoder_self_attention_mask.to(device), 
                                    decoder_self_attention_mask.to(device), 
                                    decoder_cross_attention_mask.to(device),
                                    enc_start_token=False,
                                    enc_end_token=False,
                                    dec_start_token=True,
                                    dec_end_token=True)
    # labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
    # loss = criterian(
    #     kn_predictions.view(-1, en_vocab_size).to(device),
    #     labels.view(-1).to(device)
    # ).to(device)
    # valid_indicies = torch.where(labels.view(-1) == english_to_index[PADDING_TOKEN], False, True)
    # loss = loss.sum() / valid_indicies.sum()
    # losses.append(loss)
    # epoch_loss += loss.item()
    # loss.backward()
    # optim.step()
# avg_loss = epoch_loss / len(iterator)
# Append the average training loss for the epoch to the list
    # train_losses.append(loss.item())
    for i in range(5):
        kn_sentence_predicted = torch.argmax(kn_predictions[i], axis=1)
        predicted_sentence = ""
        for idx in kn_sentence_predicted:
            if idx == english_to_index[END_TOKEN]:
                break
            predicted_sentence += index_to_english[idx.item()]
        
        predicted_captions.append(predicted_sentence)
    # if batch_num % 100 == 0:
    #     print(f"Iteration {batch_num} : {loss.item()}")
    #     # print(f"image tensor: {eng_batch[0]}")
    #     print(f"Original caption: {kn_batch[0]}")
    #     kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
    #     predicted_sentence = ""
    #     for idx in kn_sentence_predicted:
    #       if idx == english_to_index[END_TOKEN]:
    #         break
    #       predicted_sentence += index_to_english[idx.item()]
    #     print(f"Predicted caption: {predicted_sentence}")


        # transformer.eval()
        # kn_sentence = ("",)
        # eng_sentence = ("should we go to the mall?",)
        # for word_counter in range(max_sequence_length):
        #     encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence)
        #     predictions = transformer(eng_sentence,
        #                               kn_sentence,
        #                               encoder_self_attention_mask.to(device), 
        #                               decoder_self_attention_mask.to(device), 
        #                               decoder_cross_attention_mask.to(device),
        #                               enc_start_token=False,
        #                               enc_end_token=False,
        #                               dec_start_token=True,
        #                               dec_end_token=False)
        #     next_token_prob_distribution = predictions[0][word_counter] # not actual probs
        #     next_token_index = torch.argmax(next_token_prob_distribution).item()
        #     next_token = index_to_kannada[next_token_index]
        #     kn_sentence = (kn_sentence[0] + next_token, )
        #     if next_token == END_TOKEN:
        #       break
        
        # print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")

with open('test_org_text_captions.txt', 'w') as f1:
    for c in captions:
        f1.write(c+'\n')
        
with open('test_Predicted_text_captions.txt', 'w') as f2:
    for c in predicted_captions:
        f2.write(c+'\n')
        
print("******DONE*******")