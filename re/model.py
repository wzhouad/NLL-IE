import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)


class NLLModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(args.n_model)]
        self.loss_fnt = nn.CrossEntropyLoss()
        for i in range(args.n_model):
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, labels=None):
        num_models = len(self.models)
        outputs = []
        for i in range(num_models):
            output = self.models[i](
                input_ids=input_ids.to(self.device[i]),
                attention_mask=attention_mask.to(self.device[i]),
                labels=labels.to(self.device[i]) if labels is not None else None,
                return_dict=False,
            )
            output = tuple([o.to(0) for o in output])
            outputs.append(output)

        model_output = outputs[-1]
        if labels is not None:
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [output[1] for output in outputs]
            probs = [F.softmax(logit, dim=-1) for logit in logits]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            reg_loss = sum([kl_div(avg_prob, prob) for prob in probs]) / num_models
            loss = loss + self.args.alpha_t * reg_loss.mean()
            model_output = (loss,) + model_output[1:] + (reg_loss,)
        return model_output

    def resize_token_embeddings(self, n):
        for i in range(len(self.models)):
            self.models[i].resize_token_embeddings(n)
