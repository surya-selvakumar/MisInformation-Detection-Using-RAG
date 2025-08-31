from typing import Optional, Dict, Any
import torch
from transformers import Trainer

LABEL_START = "<LABEL>"
LABEL_END = "</LABEL>"

class WeightedLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()

        # weight label tokens only (between LABEL tags)
        label_mask = (target != -100).clone().int()
        # crude tag detection on the fly
        # NOTE: this is approximate; in practice you may pre-mark token ranges.
        # For simplicity: boost loss on tokens that are inside <LABEL>...</LABEL>
        bs, T = target.size()
        mask = torch.zeros_like(target, dtype=torch.float)
        # find the token ids of start/end markers
        start_id = self.tokenizer.convert_tokens_to_ids(LABEL_START)
        end_id   = self.tokenizer.convert_tokens_to_ids(LABEL_END)
        # fallbacks if not present in vocab
        for b in range(bs):
            in_span = False
            for t in range(T):
                if target[b, t].item() == start_id: in_span = True
                elif target[b, t].item() == end_id: in_span = False
                if in_span: mask[b, t] = 1.0
        weights = 1.0 + 2.0*mask  # upweight label tokens x3

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))
        loss = (loss * weights.view(-1)).mean()
        return (loss, outputs) if return_outputs else loss
