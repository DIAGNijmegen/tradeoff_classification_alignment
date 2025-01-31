from transformers import RobertaModel
from torch import nn




class RobertaSequenceClassification(nn.Module):
    def __init__(self, base_model_path, num_labels,freeze_base_params=True):
        super().__init__()
        self.num_labels = num_labels
        self.freeze_base_params = freeze_base_params

        # Load the pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained(base_model_path)
        # Freeze the parameters of RoBERTa
        if freeze_base_params:
            for param in self.roberta.parameters():
                param.requires_grad = False
       

        # Create a classification head
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Use the output of the [CLS] token (first token) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classification_head(cls_output)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits, cls_output)  # Return a tuple with loss, logits, and cls_output

        return (logits, cls_output)  # Return a tuple with logits and cls_output
    

    
class ViTForImageClassificationWithEmbeddings(nn.Module):
    def __init__(self, vit_model):
        super(ViTForImageClassificationWithEmbeddings, self).__init__()
        self.vit_model = vit_model

     
    def forward(self, pixel_values):
        outputs = self.vit_model.vit(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last hidden state
        class_token_embedding = hidden_states[:, 0, :]  # Class token embedding
        logits = self.vit_model.classifier(class_token_embedding)
        return logits, class_token_embedding