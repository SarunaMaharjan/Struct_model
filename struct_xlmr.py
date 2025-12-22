import torch
from torch import nn
import torch.nn.functional as F
from transformers import XLMRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaOutput, RobertaIntermediate, RobertaSelfOutput

class StructInformedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Per-head gate, initialized to a small value for stability
        self.struct_gate = nn.Parameter(torch.zeros(self.num_attention_heads) + 0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, struct_attn_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        if struct_attn_mask is not None:
            # Reshape gate for broadcasting: [1, num_heads, 1, 1]
            gate = self.struct_gate.view(1, -1, 1, 1)
            # Apply the gate to the structural bias before adding it
            attention_scores = attention_scores + (struct_attn_mask.unsqueeze(1) * gate)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return (context_layer,)

class StructInformedRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = StructInformedAttention(config)
        self.attention_output = RobertaSelfOutput(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None, struct_attn_mask=None):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            struct_attn_mask=struct_attn_mask,
        )
        attention_output = self.attention_output(self_attention_outputs[0], hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,)

class CNNParser(nn.Module):
    def __init__(self, hidden_size, num_parser_layers=3, kernel_size=3):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(num_parser_layers)
        ])
        self.head_predictor = nn.Linear(hidden_size, hidden_size)
        self.child_predictor = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        x = hidden_states.transpose(1, 2)
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        x = x.transpose(1, 2)
        head_rep = self.head_predictor(x)
        child_rep = self.child_predictor(x)
        dependency_scores = torch.einsum('bic,bjc->bij', child_rep, head_rep)
        # Return raw scores instead of log_softmax
        return dependency_scores

class StructXLMRoberta(nn.Module):
    def __init__(self, model_name="xlm-roberta-base-local", num_front_layers=6):
        super().__init__()
        self.base_model = XLMRobertaModel.from_pretrained(model_name)
        self.config = self.base_model.config
        self.embeddings = self.base_model.embeddings
        all_layers = self.base_model.encoder.layer
        self.front_layers = nn.ModuleList(all_layers[:num_front_layers])
        
        self.rear_layers = nn.ModuleList()
        for i in range(num_front_layers, len(all_layers)):
            original_layer = all_layers[i]
            new_layer = StructInformedRobertaLayer(self.config)
            
            # Load weights, ignoring the new 'struct_gate' with strict=False
            new_layer.attention.load_state_dict(original_layer.attention.self.state_dict(), strict=False)
            new_layer.attention_output.load_state_dict(original_layer.attention.output.state_dict())
            new_layer.intermediate.load_state_dict(original_layer.intermediate.state_dict())
            new_layer.output.load_state_dict(original_layer.output.state_dict())
            self.rear_layers.append(new_layer)

        self.parser = CNNParser(self.config.hidden_size)
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, targets, pos=None, attention_mask=None):
        if attention_mask is None:
             attention_mask = (input_ids != self.config.pad_token_id).float()
        
        extended_attention_mask = self.base_model.get_extended_attention_mask(
            attention_mask, input_ids.shape, device=input_ids.device
        )
        
        hidden_states = self.embeddings(input_ids=input_ids)
        for layer in self.front_layers:
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
        
        struct_scores = self.parser(hidden_states)
        
        rear_hidden_states = hidden_states
        for layer in self.rear_layers:
            layer_outputs = layer(
                rear_hidden_states,
                attention_mask=extended_attention_mask,
                struct_attn_mask=struct_scores,
            )
            rear_hidden_states = layer_outputs[0]
        prediction_scores = self.output_layer(rear_hidden_states)
        masked_lm_loss = self.criterion(
            prediction_scores.view(-1, self.config.vocab_size),
            targets.view(-1)
        )
        return masked_lm_loss, {'struct_scores': struct_scores}