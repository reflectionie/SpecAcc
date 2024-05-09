import random
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from .hydra_heads import HydraMLP, HydraPrefixMLP, HydraCrossAttentionDecoderLayer, EagleAttentionDecoderLayer
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .hydra_choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from collections import Counter

import torch.nn.functional as F


class HydraConfig(PretrainedConfig):
    """
    Configuration class for Hydra model.

    Args:
        hydra_num_heads (int, optional): Number of heads for the Hydra layer. Default is 2.
        hydra_num_layers (int, optional): Number of Hydra layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        hydra_num_heads=4,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        grounded_heads=False,
        hidden_state_offset=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hydra_num_heads = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.hydra_head_arch = hydra_head_arch
        self.base_model_name_or_path = base_model_name_or_path
        self.grounded_heads = grounded_heads
        self.hidden_state_offset = hidden_state_offset


class HydraModel(nn.Module):
    """The Hydra Language Model Head.

    This module creates a series of prediction heads (based on the 'hydra' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        hydra_num_heads=4,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        grounded_heads=False,
        hidden_state_offset=0,
        dropout_rate=0.0,
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            hydra_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            hydra_num_layers (int, optional): Number of ResBlock layers for each Hydra head. Defaults to 0.
        """
        super().__init__()

        # Original model setup
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.orig_lm_head = base_model.lm_head
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path)

        # Hydra setup
        self.hydra = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.hydra_head_arch = hydra_head_arch
        self.hidden_state_offset = hidden_state_offset
        self.dropout_rate = dropout_rate
        self.grounded_heads = grounded_heads

        if self.hydra_head_arch == "mlp":
            self.hydra_head = HydraMLP(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head_init_weight=base_model.lm_head.weight.data
            )
            self.hydra_lm_head = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False)
        elif self.hydra_head_arch == "prefix-mlp":
            self.hydra_head = HydraPrefixMLP(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head_init_weight=base_model.lm_head.weight.data,
                dropout_rate=self.dropout_rate,
            )
            self.hydra_lm_head = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False)
        elif self.hydra_head_arch == "cross-attn":
            self.hydra_head = HydraCrossAttentionDecoderLayer(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head=self.base_model.lm_head,
            )
        elif self.hydra_head_arch == "eagle-attn":
            self.hydra_head = EagleAttentionDecoderLayer(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head=self.base_model.lm_head,
            )
        else:
            raise NotImplementedError(
                f"Hydra head architecture {self.hydra_head_arch} not supported."
            )

        # Ensure hydra_head's dtype and device align with the base_model
        self.hydra_head.to(self.base_model.dtype).to(self.base_model.device)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    # TODO (ZACK): Figure out if hydra_num_heads should just be loaded from the config
    @classmethod
    def from_pretrained(
        cls,
        hydra_head_name_or_path,
        base_model=None,
        hydra_num_heads=None,
        **kwargs,
    ):
        """
        Args:
            hydra_head_name_or_path (str): Name or path of the Hydra head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            HydraModel: A HydraModel instance loaded from the given path.
        """
        hydra_config = HydraConfig.from_pretrained(hydra_head_name_or_path)
        if hydra_num_heads is not None:
            print("Overriding hydra_num_heads as:", hydra_num_heads)
            hydra_config.hydra_num_heads = hydra_num_heads
        if base_model is not None:
            print("Overriding base_model as:", base_model)
            hydra_config.base_model_name_or_path = base_model
        base_model = KVLlamaForCausalLM.from_pretrained(
            hydra_config.base_model_name_or_path, **kwargs
        )

        model = cls(
            base_model,
            hydra_config.hydra_num_heads,
            hydra_config.hydra_num_layers,
            hydra_config.hydra_head_arch,
            hydra_config.base_model_name_or_path,
            hydra_config.grounded_heads,
            hydra_config.hidden_state_offset,
        )
        hydra_head_path = os.path.join(
            hydra_head_name_or_path, "hydra_lm_head.pt")
        if os.path.exists(hydra_head_path):
            filename = hydra_head_path
        else:
            filename = hf_hub_download(
                hydra_head_name_or_path, "hydra_lm_head.pt")
        hydra_head_state_dict = torch.load(
            filename, map_location=base_model.device)
        model.hydra_head.load_state_dict(hydra_head_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        run_hydra_head=False,
        base_hidden_states=None,
        noise_alpha=0.0,
        ensemble_hydra=False,
        base_model=None
    ):
        """Forward pass of the HydraModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Hydra heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if base_hidden_states is not None:
            with torch.inference_mode():
                outputs = None
                if output_orig:
                    orig_logits = self.orig_lm_head(base_hidden_states)
        else:
            with torch.inference_mode():
                # Pass input through the base model
                if ensemble_hydra:
                    assert base_model is not None, "enseable hydra requires base model is independent"
                    outputs = base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        output_hidden_states=self.hidden_state_offset != 0,
                    )
                    if output_orig:
                        orig_logits = base_model.lm_head(outputs[0])
                else:
                    outputs = self.base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        output_hidden_states=self.hidden_state_offset != 0,
                    )

                    if output_orig:
                        orig_logits = self.base_model.lm_head(outputs[0])

            # Clone the output hidden states
            if self.hidden_state_offset == 0:
                base_hidden_states = outputs[0].clone()
            else:
                base_hidden_states = outputs[1][-(
                    self.hidden_state_offset + 1)].clone()

        # Hydra heads only queried in model forward during training
        if not run_hydra_head:
            assert output_orig, "Must output original predictions if not running Hydra head."
            return None, outputs, orig_logits, base_hidden_states

        # From NEFT-tune
        model_dim = base_hidden_states.shape[-1]
        seq_len = (input_ids != self.tokenizer.pad_token_id).sum(
            dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(2)
        denom = torch.sqrt(seq_len * model_dim)

        noise = (torch.rand_like(base_hidden_states)
                 * 2 - 1) * noise_alpha / denom
        noise = noise.to(base_hidden_states.dtype)
        input_base_hidden_states = base_hidden_states + noise

        if self.hydra_head_arch == "mlp":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                base_hidden_states=input_base_hidden_states, input_ids=input_ids, noise=noise
            )
        elif self.hydra_head_arch == "prefix-mlp":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                base_hidden_states=base_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                noise=noise,
            )
        elif self.hydra_head_arch == "cross-attn":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                input_ids=input_ids,
                base_hidden_states=input_base_hidden_states,
                forward_mode="training",
                base_hidden_states_position_ids=position_ids,
                attention_mask=attention_mask,
                noise=noise,
            )
            # So that they can be stacked
            hydra_logits = [hydra_logits]
            hydra_hidden_states = [hydra_hidden_states]
        elif self.hydra_head_arch == "eagle-attn":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                input_ids=input_ids,
                base_hidden_states=input_base_hidden_states,
                forward_mode="training",
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # So that they can be stacked
            hydra_logits = [hydra_logits]
            hydra_hidden_states = [hydra_hidden_states]

        if output_orig:
            return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs, orig_logits, base_hidden_states
        return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs

    def hydra_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Hydra
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        hydra_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Hydra output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            hydra_choices (list, optional): A list of integers indicating the number of choices for each Hydra head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache hydra buffers (the fixed patterns for tree attention)
        if hasattr(self, "hydra_choices") and self.hydra_choices == hydra_choices:
            # Load the cached hydra buffer
            hydra_buffers = self.hydra_buffers
        else:
            # Initialize the hydra buffer
            hydra_buffers = generate_hydra_buffers(
                hydra_choices, device=self.base_model.device
            )
        self.hydra_buffers = hydra_buffers
        self.hydra_choices = hydra_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.hydra_head_arch)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_hydra_mode(self)
        # Initialize tree attention mask and process prefill tokens
        hidden_states, logits = initialize_hydra(
            input_ids, self, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
        )

        new_token = 0
        last_round_token = 0
        total_accept = 0
        for idx in range(max_steps):
            # Generate candidates with topk predictions from Hydra heads
            to_pass_input_ids = None
            if idx == 0:
                to_pass_input_ids = input_ids
            candidates, tree_candidates = self.hydra_head.proposal(
                logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)

            # Use tree attention to verify the candidates and get predictions
            hidden_states, logits = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers[
                    "max_accepts"]
            )
            total_accept = accept_length + total_accept

            # Update the input_ids and logits
            input_ids, logits, hidden_states, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                hydra_buffers["retrieve_indices"],
                logits,
                hidden_states,
                new_token,
                past_key_values_data,
                current_length_data,
                self.hydra_head_arch,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

    def hydra_generate_reflectio(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Hydra
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        hydra_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Hydra output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,


        max_length=2048,
        max_new_tokens=512
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            hydra_choices (list, optional): A list of integers indicating the number of choices for each Hydra head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache hydra buffers (the fixed patterns for tree attention)
        if hasattr(self, "hydra_choices") and self.hydra_choices == hydra_choices:
            # Load the cached hydra buffer
            hydra_buffers = self.hydra_buffers
        else:
            # Initialize the hydra buffer
            hydra_buffers = generate_hydra_buffers(
                hydra_choices, device=self.base_model.device
            )
        self.hydra_buffers = hydra_buffers
        self.hydra_choices = hydra_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.hydra_head_arch)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_hydra_mode(self)
        # Initialize tree attention mask and process prefill tokens
        hidden_states, logits = initialize_hydra(
            input_ids, self, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
        )

        new_token = 0
        last_round_token = 0
        total_accept = 0

        accept_token_length = []

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Hydra heads
            to_pass_input_ids = None
            if idx == 0:
                to_pass_input_ids = input_ids
            candidates, tree_candidates = self.hydra_head.proposal(
                logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)

            # Use tree attention to verify the candidates and get predictions
            hidden_states, logits = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers[
                    "max_accepts"]
            )
            total_accept = accept_length + total_accept

            # Update the input_ids and logits
            input_ids, logits, hidden_states, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                hydra_buffers["retrieve_indices"],
                logits,
                hidden_states,
                new_token,
                past_key_values_data,
                current_length_data,
                self.hydra_head_arch,
            )

            accept_token_length.append(int(accept_length))

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids, accept_token_length
            if new_token > max_new_tokens:
                return input_ids, accept_token_length
            if input_ids.shape[1] > max_length:
                return input_ids, accept_token_length


# ============================================================================================

class HydraModelEnsembleHead(nn.Module):
    """The Hydra Language Model Head.

    This module creates a series of prediction heads (based on the 'hydra' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        hydra_num_heads=4,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        grounded_heads=False,
        hidden_state_offset=0,
        dropout_rate=0.0,

    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            hydra_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            hydra_num_layers (int, optional): Number of ResBlock layers for each Hydra head. Defaults to 0.
        """
        super().__init__()

        # Original model setup
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.orig_lm_head = base_model.lm_head
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path)

        # Hydra setup
        self.hydra = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.hydra_head_arch = hydra_head_arch
        self.hidden_state_offset = hidden_state_offset
        self.dropout_rate = dropout_rate
        self.grounded_heads = grounded_heads

        if self.hydra_head_arch == "mlp":
            self.hydra_head = HydraMLP(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head_init_weight=base_model.lm_head.weight.data
            )
            self.hydra_lm_head = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False)
        elif self.hydra_head_arch == "prefix-mlp":
            self.hydra_head = HydraPrefixMLP(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head_init_weight=base_model.lm_head.weight.data,
                dropout_rate=self.dropout_rate,
            )
            self.hydra_lm_head = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False)
        elif self.hydra_head_arch == "cross-attn":
            self.hydra_head = HydraCrossAttentionDecoderLayer(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head=self.base_model.lm_head,
            )
        elif self.hydra_head_arch == "eagle-attn":
            self.hydra_head = EagleAttentionDecoderLayer(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.base_model.model.embed_tokens,
                base_config=self.config,
                lm_head=self.base_model.lm_head,
            )
        else:
            raise NotImplementedError(
                f"Hydra head architecture {self.hydra_head_arch} not supported."
            )

        # Ensure hydra_head's dtype and device align with the base_model
        self.hydra_head.to(self.base_model.dtype).to(self.base_model.device)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    # TODO (ZACK): Figure out if hydra_num_heads should just be loaded from the config
    @classmethod
    def from_pretrained(
        cls,
        hydra_head_name_or_path,
        base_model=None,
        hydra_num_heads=None,
        **kwargs,
    ):
        """
        Args:
            hydra_head_name_or_path (str): Name or path of the Hydra head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            HydraModel: A HydraModel instance loaded from the given path.
        """
        hydra_config = HydraConfig.from_pretrained(hydra_head_name_or_path)
        if hydra_num_heads is not None:
            print("Overriding hydra_num_heads as:", hydra_num_heads)
            hydra_config.hydra_num_heads = hydra_num_heads
        if base_model is not None:
            print("Overriding base_model as:", base_model)
            hydra_config.base_model_name_or_path = base_model.name_or_path
        # base_model = KVLlamaForCausalLM.from_pretrained(
        #     hydra_config.base_model_name_or_path, **kwargs
        # )

        model = cls(
            base_model,
            hydra_config.hydra_num_heads,
            hydra_config.hydra_num_layers,
            hydra_config.hydra_head_arch,
            hydra_config.base_model_name_or_path,
            hydra_config.grounded_heads,
            hydra_config.hidden_state_offset,
        )
        hydra_head_path = os.path.join(
            hydra_head_name_or_path, "hydra_lm_head.pt")
        if os.path.exists(hydra_head_path):
            filename = hydra_head_path
        else:
            filename = hf_hub_download(
                hydra_head_name_or_path, "hydra_lm_head.pt")
        hydra_head_state_dict = torch.load(
            filename, map_location=base_model.device)
        model.hydra_head.load_state_dict(hydra_head_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        run_hydra_head=False,
        base_hidden_states=None,
        noise_alpha=0.0,
        ensemble_hydra=False,
        base_model=None
    ):
        """Forward pass of the HydraModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Hydra heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if base_hidden_states is not None:
            with torch.inference_mode():
                outputs = None
                if output_orig:
                    orig_logits = self.orig_lm_head(base_hidden_states)
        else:
            with torch.inference_mode():
                # Pass input through the base model
                if ensemble_hydra:
                    assert base_model is not None, "enseable hydra requires base model is independent"
                    outputs = base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        output_hidden_states=self.hidden_state_offset != 0,
                    )
                    if output_orig:
                        orig_logits = base_model.lm_head(outputs[0])
                else:
                    outputs = self.base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        output_hidden_states=self.hidden_state_offset != 0,
                    )

                    if output_orig:
                        orig_logits = self.base_model.lm_head(outputs[0])

            # Clone the output hidden states
            if self.hidden_state_offset == 0:
                base_hidden_states = outputs[0].clone()
            else:
                base_hidden_states = outputs[1][-(
                    self.hidden_state_offset + 1)].clone()

        # Hydra heads only queried in model forward during training
        if not run_hydra_head:
            assert output_orig, "Must output original predictions if not running Hydra head."
            return None, outputs, orig_logits, base_hidden_states

        # From NEFT-tune
        model_dim = base_hidden_states.shape[-1]
        seq_len = (input_ids != self.tokenizer.pad_token_id).sum(
            dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(2)
        denom = torch.sqrt(seq_len * model_dim)

        noise = (torch.rand_like(base_hidden_states)
                 * 2 - 1) * noise_alpha / denom
        noise = noise.to(base_hidden_states.dtype)
        input_base_hidden_states = base_hidden_states + noise

        if self.hydra_head_arch == "mlp":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                base_hidden_states=input_base_hidden_states, input_ids=input_ids, noise=noise
            )
        elif self.hydra_head_arch == "prefix-mlp":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                base_hidden_states=base_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                noise=noise,
            )
        elif self.hydra_head_arch == "cross-attn":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                input_ids=input_ids,
                base_hidden_states=input_base_hidden_states,
                forward_mode="training",
                base_hidden_states_position_ids=position_ids,
                attention_mask=attention_mask,
                noise=noise,
            )
            # So that they can be stacked
            hydra_logits = [hydra_logits]
            hydra_hidden_states = [hydra_hidden_states]
        elif self.hydra_head_arch == "eagle-attn":
            hydra_logits, hydra_hidden_states = self.hydra_head(
                input_ids=input_ids,
                base_hidden_states=input_base_hidden_states,
                forward_mode="training",
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # So that they can be stacked
            hydra_logits = [hydra_logits]
            hydra_hidden_states = [hydra_hidden_states]

        if output_orig:
            return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs, orig_logits, base_hidden_states
        return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs

    def hydra_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Hydra
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        hydra_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Hydra output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            hydra_choices (list, optional): A list of integers indicating the number of choices for each Hydra head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache hydra buffers (the fixed patterns for tree attention)
        if hasattr(self, "hydra_choices") and self.hydra_choices == hydra_choices:
            # Load the cached hydra buffer
            hydra_buffers = self.hydra_buffers
        else:
            # Initialize the hydra buffer
            hydra_buffers = generate_hydra_buffers(
                hydra_choices, device=self.base_model.device
            )
        self.hydra_buffers = hydra_buffers
        self.hydra_choices = hydra_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.hydra_head_arch)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_hydra_mode(self)
        # Initialize tree attention mask and process prefill tokens
        hidden_states, logits = initialize_hydra(
            input_ids, self, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
        )

        new_token = 0
        last_round_token = 0
        total_accept = 0
        for idx in range(max_steps):
            # Generate candidates with topk predictions from Hydra heads
            to_pass_input_ids = None
            if idx == 0:
                to_pass_input_ids = input_ids
            candidates, tree_candidates = self.hydra_head.proposal(
                logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)

            # Use tree attention to verify the candidates and get predictions
            hidden_states, logits = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers[
                    "max_accepts"]
            )
            total_accept = accept_length + total_accept

            # Update the input_ids and logits
            input_ids, logits, hidden_states, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                hydra_buffers["retrieve_indices"],
                logits,
                hidden_states,
                new_token,
                past_key_values_data,
                current_length_data,
                self.hydra_head_arch,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

    def hydra_generate_reflectio(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Hydra
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        hydra_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Hydra output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,


        max_length=2048,
        max_new_tokens=512
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            hydra_choices (list, optional): A list of integers indicating the number of choices for each Hydra head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache hydra buffers (the fixed patterns for tree attention)
        if hasattr(self, "hydra_choices") and self.hydra_choices == hydra_choices:
            # Load the cached hydra buffer
            hydra_buffers = self.hydra_buffers
        else:
            # Initialize the hydra buffer
            hydra_buffers = generate_hydra_buffers(
                hydra_choices, device=self.base_model.device
            )
        self.hydra_buffers = hydra_buffers
        self.hydra_choices = hydra_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.hydra_head_arch)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_hydra_mode(self)
        # Initialize tree attention mask and process prefill tokens
        hidden_states, logits = initialize_hydra(
            input_ids, self, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
        )

        new_token = 0
        last_round_token = 0
        total_accept = 0

        accept_token_length = []

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Hydra heads
            to_pass_input_ids = None
            if idx == 0:
                to_pass_input_ids = input_ids
            candidates, tree_candidates = self.hydra_head.proposal(
                logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)

            # Use tree attention to verify the candidates and get predictions
            hidden_states, logits = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers[
                    "max_accepts"]
            )
            total_accept = accept_length + total_accept

            # Update the input_ids and logits
            input_ids, logits, hidden_states, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                hydra_buffers["retrieve_indices"],
                logits,
                hidden_states,
                new_token,
                past_key_values_data,
                current_length_data,
                self.hydra_head_arch,
            )

            accept_token_length.append(int(accept_length))

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids, accept_token_length
            if new_token > max_new_tokens:
                return input_ids, accept_token_length
            if input_ids.shape[1] > max_length:
                return input_ids, accept_token_length

# ============================================================


def ensemble_method(hydra_candidates_ensemble):
    # 初始假设第一个元素为最大值
    hydra_candidates = hydra_candidates_ensemble[0]
    for tensor in hydra_candidates_ensemble:
        # 比较当前张量和目前最大张量的值
        if tensor > hydra_candidates:
            hydra_candidates = tensor
    return hydra_candidates


def top_k_ensemble(hydra_candidates_ensemble, hydra_candidates_values_ensemble, topK):
    # 确保输入是正确的
    assert len(hydra_candidates_ensemble) == len(
        hydra_candidates_values_ensemble)

    # 使用字典来累加相同元素的概率
    candidate_probabilities = {}

    # 遍历所有的tensor对
    for candidates, logits in zip(hydra_candidates_ensemble, hydra_candidates_values_ensemble):
        probs = F.softmax(logits, dim=1)
        for candidate, logit in zip(candidates.squeeze(0), probs.squeeze(0)):
            candidate = candidate.item()  # 转换为Python数值
            if candidate in candidate_probabilities:
                candidate_probabilities[candidate] += logit.item()
            else:
                candidate_probabilities[candidate] = logit.item()

    # 排序并选取概率最高的10个元素
    sorted_candidates = sorted(
        candidate_probabilities.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:topK]

    # 构造最终的tensor
    final_candidates = [candidate for candidate, _ in top_candidates]
    final_candidates = torch.tensor(final_candidates).view(1, topK)

    return final_candidates


def proposal(ensemble_model,
             input_logits,
             base_hidden_states,
             #  hydra_buffers,
             #  past_key_values
             ):
    """
    生成ensemble的logits
    """
    children_per_head = ensemble_model[0].hydra_buffers["children_per_head"]
    children_to_expand_per_head = ensemble_model[0].hydra_buffers["children_to_expand_per_head"]
    retrieve_indices = ensemble_model[0].hydra_buffers["retrieve_indices"]

    # Build prefix through attn layer
    # Fixed to only one layer currently

    candidates_ensemble = []
    candidates_embeddings_ensemble = []

    ensemble_model[0].past_key_values = ensemble_model[0].past_key_values[-1:]
    past_seq_len = ensemble_model[0].past_key_values[0][0].current_length
    seq_len = past_seq_len + base_hidden_states.shape[1]
    position_ids = torch.arange(
        past_seq_len, seq_len, device=input_logits.device).unsqueeze(0)
    for idx, hydra_module in enumerate(ensemble_model):
        # hydra_module.past_key_values = hydra_module.past_key_values[-1:]
        # past_seq_len = hydra_module.past_key_values[0][0].current_length
        # seq_len = past_seq_len + base_hidden_states_list[idx].shape[1]
        # position_ids = torch.arange(
        #     past_seq_len, seq_len, device=input_logits.device).unsqueeze(0)
        prefix_embedding = hydra_module.hydra_head.prefix_embeding_layer(
            inputs_embeds=base_hidden_states,
            attention_mask=None,  # Might need to change eventually
            past_key_values=hydra_module.past_key_values,
            position_ids=position_ids,
        )[0]

        candidate_id = torch.argmax(input_logits[:, -1]).unsqueeze(0)
        candidate_embedding = hydra_module.hydra_head.input_embed_fn(
            candidate_id).unsqueeze(0)

        candidates = torch.tensor(
            [candidate_id], device=candidate_id.device)[None, ...]
        candidates_embeddings = torch.cat(
            [prefix_embedding[:, -1:], candidate_embedding], dim=-1)

        candidates_ensemble.append(candidates)
        candidates_embeddings_ensemble.append(candidates_embeddings)
    # 对candidates进行投票，得到ensemble后的candidates
    candidates = ensemble_method(candidates_ensemble)

    for head_idx, (head_num_children, head_children_to_expand) in enumerate(zip(children_per_head, children_to_expand_per_head)):

        hydra_preds_ensemble = []
        for idx, hydra_module in enumerate(ensemble_model):
            hydra_hidden_state = hydra_module.hydra_head.hydra_mlp[head_idx](
                candidates_embeddings_ensemble[idx])
            hydra_preds = hydra_module.hydra_head.hydra_lm_head[head_idx](
                hydra_hidden_state)
            hydra_preds_ensemble.append(hydra_preds)

        next_head_embeddings = []

        # hydra_hidden_state = self.hydra_mlp[head_idx](candidates_embeddings)
        # hydra_preds = self.hydra_lm_head[head_idx](hydra_hidden_state)
        # next_head_embeddings = []
        hydra_candidates_ensemble = []
        hydra_candidates_values_ensemble = []

        next_head_embeddings_ensemble = []
        for path_idx, (num_children, children_to_expand) in enumerate(zip(head_num_children, head_children_to_expand)):
            # hydra_candidates = torch.topk(hydra_preds[:, path_idx], num_children, dim=-1).indices
            # candidates = torch.cat([candidates, hydra_candidates], dim=-1)
            for idx, hydra_module in enumerate(ensemble_model):

                # TODO: top_k_ensemble
                hydra_candidates = torch.topk(
                    hydra_preds_ensemble[idx][:, path_idx], num_children, dim=-1).indices
                hydra_candidates_values = torch.topk(
                    hydra_preds_ensemble[idx][:, path_idx], num_children, dim=-1).values
                hydra_candidates_ensemble.append(hydra_candidates)
                hydra_candidates_values_ensemble.append(
                    hydra_candidates_values)
            # 对 hydra_candidates进行投票，得到ensemble后的hydra_candidates
            hydra_candidates = top_k_ensemble(
                hydra_candidates_ensemble, hydra_candidates_values_ensemble, topK=num_children).to(candidates)

            candidates = torch.cat([candidates, hydra_candidates], dim=-1)

            # next_head_embeddings_ensemble = []
            if children_to_expand > 0:
                for idx, hydra_module in enumerate(ensemble_model):
                    children_embeddings = hydra_module.hydra_head.input_embed_fn(
                        hydra_candidates)[:, :children_to_expand]
                    repeat_slice = [path_idx] * children_to_expand
                    path_embeddings = candidates_embeddings_ensemble[idx][:, repeat_slice]
                    next_head_embeddings.append(
                        torch.cat([path_embeddings, children_embeddings], dim=-1))
                    next_head_embeddings_ensemble.append(next_head_embeddings)

                # children_embeddings = self.input_embed_fn(hydra_candidates)[:, :children_to_expand]
                # repeat_slice = [path_idx] * children_to_expand
                # path_embeddings = candidates_embeddings[:, repeat_slice]
                # next_head_embeddings.append(torch.cat([path_embeddings, children_embeddings], dim=-1))

        if len(next_head_embeddings_ensemble):
            for idx, hydra_module in enumerate(ensemble_model):
                candidates_embeddings_ensemble[idx] = torch.cat(
                    next_head_embeddings_ensemble[idx], dim=1)
            # TODO (Zack): Determine assertion error about next_head_embeddings being empty before finishing tree
            # candidates_embeddings = torch.cat(next_head_embeddings, dim=1)

    # TODO (Zack): Only selecting first batch element for now, change when doing bs > 1
    cart_candidates = candidates[0, retrieve_indices]

    return cart_candidates, candidates


def ensemble_hydra_generate_reflectio(
    model,
    input_ids,
    temperature=0,
    max_new_tokens=512,
    hydra_choices=mc_sim_7b_63,
    max_steps=512
):

    base_model = model['base_model']
    ensemble_model = model['ensemble_model']

    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    # Cache hydra buffers (the fixed patterns for tree attention)
    if hasattr(ensemble_model[0], "hydra_choices") and ensemble_model[0].hydra_choices == hydra_choices:
        # Load the cached hydra buffer
        hydra_buffers = ensemble_model[0].hydra_buffers
    else:
        # Initialize the hydra buffer
        hydra_buffers = generate_hydra_buffers(
            hydra_choices, device=base_model.device
        )
    for hydra_module in ensemble_model:
        # # Cache hydra buffers (the fixed patterns for tree attention)
        # if hasattr(hydra_module, "hydra_choices") and hydra_module.hydra_choices == hydra_choices:
        #     # Load the cached hydra buffer
        #     hydra_buffers = hydra_module.hydra_buffers
        # else:
        #     # Initialize the hydra buffer
        #     hydra_buffers = generate_hydra_buffers(
        #         hydra_choices, device=base_model.device
        #     )
        hydra_module.hydra_buffers = hydra_buffers
        hydra_module.hydra_choices = hydra_choices

        # # Initialize the past key and value states
        # if hasattr(hydra_module, "past_key_values"):
        #     past_key_values = hydra_module.past_key_values
        #     past_key_values_data = hydra_module.past_key_values_data
        #     current_length_data = hydra_module.current_length_data
        #     # Reset the past key and value states
        #     current_length_data.zero_()
        # else:
        #     (
        #         past_key_values,
        #         past_key_values_data,
        #         current_length_data,
        #     ) = initialize_past_key_values(hydra_module.base_model, hydra_module.hydra_head_arch)
        #     hydra_module.past_key_values = past_key_values
        #     hydra_module.past_key_values_data = past_key_values_data
        #     hydra_module.current_length_data = current_length_data
    # Initialize the past key and value states
    if hasattr(ensemble_model[0], "past_key_values"):
        for hydra_module in ensemble_model:
            past_key_values = hydra_module.past_key_values
            past_key_values_data = hydra_module.past_key_values_data
            current_length_data = hydra_module.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(base_model, ensemble_model[0].hydra_head_arch)
        for hydra_module in ensemble_model:
            hydra_module.past_key_values = past_key_values
            hydra_module.past_key_values_data = past_key_values_data
            hydra_module.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    # reset_hydra_mode(hydra_module)
    base_model.model.hydra_mask = None
    base_model.model.hydra_mode = None
    # Initialize tree attention mask and process prefill tokens

    # hidden_states, logits = initialize_hydra(
    # input_ids, hydra_module, hydra_buffers["hydra_attn_mask"], hydra_module.past_key_values, hydra_buffers["proposal_cross_attn_masks"]
    # )
    # initialize hydra返回的东西只需要base_model的，所以调用一个头就可以

    _, outputs, orig_logits, _ = ensemble_model[0](
        input_ids,
        past_key_values=past_key_values,
        output_orig=True,
        ensemble_hydra=True,
        base_model=base_model
    )
    if ensemble_model[0].hidden_state_offset == 0:
        hidden_states = outputs[0].clone()
    else:
        hidden_states = outputs[1][-(
            ensemble_model[0].hidden_state_offset + 1)].clone()
        
    base_model.model.hydra_mask = ensemble_model[0].hydra_buffers["hydra_attn_mask"]
    
    if ensemble_model[0].hydra_head_arch == "cross-attn":
        # 这里一定是false，所以没有改
        model.hydra_head.proposal_hydra_masks = ensemble_model[
            0].hydra_buffers["proposal_cross_attn_masks"]
    # return hidden_states, logits
    logits = orig_logits

    new_token = 0
    last_round_token = 0
    total_accept = 0

    accept_token_length = []

    for idx in range(max_steps):
        # Generate candidates with topk predictions from Hydra heads
        to_pass_input_ids = None
        if idx == 0:
            to_pass_input_ids = input_ids

        candidates, tree_candidates = proposal(
            ensemble_model=ensemble_model, input_logits=logits, base_hidden_states=hidden_states)

        # Use tree attention to verify the candidates and get predictions
        hidden_states, logits = ensemble_tree_decoding(
            ensemble_model,
            base_model,
            tree_candidates,
            past_key_values,
            # hydra_buffers["hydra_position_ids"],
            input_ids,
            # hydra_buffers["retrieve_indices"],
        )

        # # Use tree attention to verify the candidates and get predictions
        # hidden_states, logits = tree_decoding(
        #     self,
        #     tree_candidates,
        #     past_key_values,
        #     hydra_buffers["hydra_position_ids"],
        #     input_ids,
        #     hydra_buffers["retrieve_indices"],
        # )

        # Evaluate the posterior of the candidates to select the accepted candidate prefix
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers[
                "max_accepts"]
        )
        total_accept = accept_length + total_accept

        # Update the input_ids and logits
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            past_key_values_data,
            current_length_data,
            self.hydra_head_arch,
        )

        accept_token_length.append(int(accept_length))

        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            return input_ids, accept_token_length
        if new_token > max_new_tokens:
            return input_ids, accept_token_length
        if input_ids.shape[1] > max_length:
            return input_ids, accept_token_length

    pass
