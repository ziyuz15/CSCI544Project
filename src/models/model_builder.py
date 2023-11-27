import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.init import xavier_uniform_

from models.encoder import Classifier
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        # encoded_layers, _ = self.model(x, segs, attention_mask =mask)
        # top_vec = encoded_layers[-1]
        # Example: Using a combination of last four layers
        encoded_layers = self.model(x, segs, attention_mask=mask)[0]
        top_vec = torch.mean(torch.stack(encoded_layers[-4:]), dim=0)

        return top_vec
# class T5Summarizer(nn.Module):
#     def __init__(self, args, device, t5_model="t5-small"):
#         super(T5Summarizer, self).__init__()
#         self.args = args
#         self.device = device
        
#         # Initialize T5 model and tokenizer
#         self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
#         self.tokenizer = T5Tokenizer.from_pretrained(t5_model)

#         self.t5 = self.t5.to(device)
        
#     def forward(self, input_text):
#     # Tokenize and prepare the input text
#     input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

#     # Generate summary with T5
#     summary_ids = self.t5.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return summary


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert=False, bert_config=None):#, t5_model="t5-small"):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        
        # self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)
        # self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        # self.t5 = self.t5.to(device)
        
        if args.encoder == 'classifier':
            self.encoder = Classifier(self.bert.model.config.hidden_size, args.dropout)
        
        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls):#, src_txt):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        
        # # Check if src_txt is None
        # if src_txt is not None:
        #     t5_input = self.prepare_t5_input(sents_vec, src_txt)
        #     t5_output = self.t5.generate(input_ids=t5_input)
        # else:
        #     # Handle case where src_txt is None
        #     t5_output = None
        
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls#, t5_output
    
    # def prepare_t5_input(self, sent_scores, src_txt, max_sentences=3):
    #     if src_txt is None:
    #         # Handle case where src_txt is None
    #         # For example, return a default value or process differently
    #         return None
    #     top_sentences_indices = torch.topk(sent_scores, k=max_sentences, dim=1).indices
    #     t5_formatted_input = []
    #     for batch_index in range(len(src_txt)):
    #         selected_sentences = [src_txt[batch_index][idx] for idx in top_sentences_indices[batch_index]]
    #         summary = ' '.join(selected_sentences)
    #         t5_formatted_input.append(summary)

    #     t5_input_ids = self.t5_tokenizer(t5_formatted_input, padding=True, truncation=True, return_tensors="pt", max_length=512)
    #     t5_input_ids = t5_input_ids.to(self.device)

    #     return t5_input_ids


# class Summarizer(nn.Module):
#     def __init__(self, args, device, load_pretrained_bert = False, bert_config = None,t5_model="t5-small"):
#         super(Summarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        
#         self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)
#         self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
#         self.t5 = self.t5.to(device)
        
#         if (args.encoder == 'classifier'):
#             # self.encoder = Classifier(self.bert.model.config.hidden_size)
#             self.encoder = Classifier(self.bert.model.config.hidden_size,args.dropout)
#         if args.param_init != 0.0:
#             for p in self.encoder.parameters():
#                 p.data.uniform_(-args.param_init, args.param_init)
#         if args.param_init_glorot:
#             for p in self.encoder.parameters():
#                 if p.dim() > 1:
#                     xavier_uniform_(p)
#         self.to(device)
#     def load_cp(self, pt):
#         self.load_state_dict(pt['model'], strict=True)

#     def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
#         top_vec = self.bert(x, segs, mask)
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#         sents_vec = sents_vec * mask_cls[:, :, None].float()
        
#         t5_input = self.prepare_t5_input(sents_vec, src_txt)
#         t5_output = self.t5.generate(input_ids=t5_input)
        
#         sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
#         return sent_scores, mask_cls
    
#     def prepare_t5_input(self, sent_scores, src_txt, max_sentences=3):
#         top_sentences_indices = torch.topk(sent_scores, k=max_sentences, dim=1).indices
#         t5_formatted_input = []
#         for batch_index in range(len(src_txt)):
#             selected_sentences = [src_txt[batch_index][idx] for idx in top_sentences_indices[batch_index]]
#             summary = ' '.join(selected_sentences)
#             t5_formatted_input.append(summary)

#         t5_input_ids = self.t5_tokenizer(t5_formatted_input, padding=True, truncation=True, return_tensors="pt", max_length=512)
#         t5_input_ids = t5_input_ids.to(self.device)

#         return t5_input_ids


