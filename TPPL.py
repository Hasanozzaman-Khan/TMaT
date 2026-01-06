
# %%
import torch
import traceback
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"


class BARTScorerSymmetricLogPPL:

    def __init__(self, device=device, max_length=1024, checkpoint="facebook/bart-base"):
        self.device = device
        self.max_length = max_length

        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()

    def load(self, path=None):
        if path is None:
            path = "models/bart.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, texts_x: List[dict], tokenized_y: dict, batch_size=4):
        scores = []

        for i in range(0, len(texts_x), batch_size):
            x_batch = texts_x[i:i + batch_size]

            try:
                with torch.no_grad():
                    x_ids = torch.stack([x["input_ids"] for x in x_batch]).to(self.device)
                    x_mask = torch.stack([x["attention_mask"] for x in x_batch]).to(self.device)

                    y_ids = tokenized_y["input_ids"].unsqueeze(0).repeat(len(x_batch), 1).to(self.device)
                    y_mask = tokenized_y["attention_mask"].unsqueeze(0).repeat(len(x_batch), 1).to(self.device)

                    assert x_ids.shape == y_ids.shape, f"x_ids {x_ids.shape} vs y_ids {y_ids.shape}"
                    assert x_mask.shape == y_mask.shape, f"x_mask {x_mask.shape} vs y_mask {y_mask.shape}"

                    ignore_idx = -100  
                    y_labels = y_ids.clone()
                    x_labels = x_ids.clone()
                    y_labels[~y_mask.bool()] = ignore_idx
                    x_labels[~x_mask.bool()] = ignore_idx

                    out_y_given_x = self.model(
                        input_ids=x_ids,
                        attention_mask=x_mask,
                        labels=y_labels
                    )

                    logp_y = torch.log_softmax(out_y_given_x.logits, dim=-1)
                    logp_y = logp_y.gather(2, y_ids.unsqueeze(-1)).squeeze(-1)
                    logp_y = logp_y * y_mask  

                    out_x_given_y = self.model(
                        input_ids=y_ids,
                        attention_mask=y_mask,
                        labels=x_labels
                    )
                    logp_x = torch.log_softmax(out_x_given_y.logits, dim=-1)
                    logp_x = logp_x.gather(2, x_ids.unsqueeze(-1)).squeeze(-1)
                    logp_x = logp_x * x_mask  

                    s_t = (logp_y + logp_x) / 2.0
                    token_lengths = x_mask.sum(dim=1).float()
                    s_bar = s_t.sum(dim=1) / token_lengths
                    batch_scores = s_bar

                    scores.extend(batch_scores.cpu().tolist())

            except RuntimeError:
                traceback.print_exc()
                print("X batch:", x_batch)
                exit(1)

        return scores
