from transformers import AutoModel, AutoConfig, Trainer
import torch
import torch.nn as nn

class PhoBERTMultiAspectModel(nn.Module):
   def __init__(self, model_name, num_aspects):
        super(PhoBERTMultiAspectModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = AutoConfig.from_pretrained(model_name)  
        self.hidden_size = self.bert.config.hidden_size * 4 

        # Create a classifier for each aspect (4 logits per aspect)
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, 4) for _ in range(num_aspects)
        ])

        self.dropout = nn.Dropout(0.2)

   def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                            token_type_ids=token_type_ids if token_type_ids is not None else None)
        
        hidden_states = outputs.hidden_states  # Shape: (batch_size, seq_len, hidden_dim)

        # Correctly concatenate the last 4 hidden layers
        # pooled_output = torch.cat([hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        # pooled_output = self.dropout(pooled_output)
        pooled_output = torch.cat([hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        pooled_output = self.dropout(pooled_output)

        # Apply classifiers to each aspect
        aspect_outputs = [classifier(pooled_output) for classifier in self.aspect_classifiers]

        # Return a structured output: (batch_size, num_aspects, 4)
        return torch.stack(aspect_outputs, dim=1)  
    
class PhoBERTTrainer(Trainer):
    def get_train_dataloader(self):
        print("✅ Using custom train DataLoader")
        return train_loader  # type: ignore # ✅ Ensures labels are included in batches

    def get_eval_dataloader(self, eval_dataset=None):
        """Forces Trainer to use the custom eval DataLoader."""
        print("✅ Using custom eval DataLoader")  
        return eval_loader  # type: ignore # ✅ Ensures validation labels are included

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss function to correctly process multi-aspect labels."""
        # print("Keys in inputs:", inputs.keys())  # Debugging print

        if "labels" not in inputs:
            raise ValueError(f"🚨 Missing 'labels' in inputs. Available keys: {inputs.keys()}")

        labels = inputs["labels"]  # ✅ Access labels without popping them
        outputs = model(**inputs)  # Forward pass → (batch_size, num_aspects, 4)

        # Convert one-hot encoded labels to class indices
        labels = torch.argmax(labels, dim=-1)  # Shape: (batch_size, num_aspects)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, 4), labels.view(-1))  # Reshape correctly for CE Loss

        return (loss, outputs) if return_outputs else loss
