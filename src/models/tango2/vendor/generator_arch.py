# vendored: declare-lab/tango c2604786e4e08e16a42232e1457b5fcd7d1c4fae (CC-BY-NC-ND-4.0)
# From model/models.py, the AudioDiffusion class only.
# Dropped build_pretrained_models: it calls get_metadata(), which hits a URL.
# Dropped compute_snr, forward, and the snr_gamma/uncondition training path.
# Dropped inference and prepare_latents: the wrapper owns the sampling loop,
# per-row seeding, and the fixed frame geometry.
# Dropped the CLIPTokenizer/CLIPTextModel/AutoModel branches: the released
# checkpoint is T5 only, so that branching is dead weight.
# Dropped the audioldm imports and wav_to_fbank: the codec is in
# tokenizer_arch.py, and vendored code imports no project code.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import UNet2DConditionModel


class AudioDiffusion(nn.Module):
    def __init__(self, text_encoder_name, unet_config):
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.unet = UNet2DConditionModel.from_config(unet_config)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask
