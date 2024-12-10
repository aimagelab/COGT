from torch import nn
import logging as python_logging
from clip_models import CLIPWrapper
import torch
from dataset import fine_graned_dependency_dictionary
from models.gpt2 import GPT2Config, GPT2LMHeadModel
import einops
from typing import Optional

python_logging.basicConfig(level=python_logging.INFO)


class CapWrapper(nn.Module):
    def __init__(self,
                 tokenizer_q_former,
                 visual_backbone: CLIPWrapper,
                 tokenizer: str,
                 args):
        """
        :param visual_backbone:
        :param encoder
        :param encoder:
        """

        super().__init__()

        self.tokenizer_q_former = tokenizer_q_former
        self.visual_backbone = visual_backbone.visual if not args.xvlm and not args.instruct_blip else visual_backbone
        self.tokenizer = tokenizer
        self.device = args.device
        self.args = args
        self.dtype = torch.float16 if self.args.fp16_enabled else torch.float32
        self.xvlm = args.xvlm
        self.instruct_blip = args.instruct_blip

        if self.xvlm:
            in_dim = visual_backbone.norm.weight.shape[0]
        elif self.instruct_blip:
            in_dim = self.visual_backbone.qformer.encoder.layer[-1].output_query.LayerNorm.weight.shape[0]
        else:
            in_dim = self.visual_backbone.conv1.weight.shape[0]

        out_dim = args.n_embd

        if self.tokenizer_q_former:
            self.prompt = self.tokenizer_q_former("Write a description for the photo.")
            self.prompt = torch.Tensor(self.prompt["input_ids"]).unsqueeze(dim=0).to(args.device).to(torch.int64)

        if not self.instruct_blip:
            self.ln_pre_out = nn.LayerNorm(in_dim).to(self.device)
        self.linear_out = nn.Linear(in_dim, out_dim).to(self.device)
        self.ln_post_out = nn.LayerNorm(out_dim).to(self.device)

        decoder_config = GPT2Config()
        decoder_config.add_cross_attention = True
        decoder_config.n_layer = args.n_layer
        decoder_config.n_embd = args.n_embd
        decoder_config.n_head = args.n_head
        decoder_config.vocab_size = tokenizer.vocab_size
        decoder_config.bos_token_id = self.tokenizer.sot_token_id
        decoder_config.eos_token_id = self.tokenizer.eot_token_id

        dependency_dictionary = fine_graned_dependency_dictionary
        decoder_config.vocab_size = tokenizer.vocab_size + len(set(dependency_dictionary.values())) + 1

        self.decoder = GPT2LMHeadModel(decoder_config)
        self.decoder.to(self.device)
        self.visual_backbone.to(self.device)

    def forward(self, pixel_values: torch.Tensor, captions: torch.Tensor, attention_mask: Optional[torch.Tensor]):

        encoder_hidden_states = None

        if self.xvlm:
            encoder_hidden_states = self.forward_xvlm(pixel_values)
        elif self.instruct_blip:
            encoder_hidden_states = self.forward_instruct_blip(pixel_values)
        else:
            self.forward_visual(pixel_values)

        output = self.decoder(
            input_ids=captions,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        return output

    def forward_visual(self, x: torch.Tensor):

        with torch.no_grad():
            x = self.visual_backbone.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            # class embeddings and positional embeddings
            x = torch.cat([_expand_token(self.visual_backbone.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
            # shape = [*, grid ** 2 + 1, width]
            x = x + self.visual_backbone.positional_embedding.to(x.dtype)

            x = self.visual_backbone.patch_dropout(x)
            x = self.visual_backbone.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND

            for r in self.visual_backbone.transformer.resblocks[:-1]:
                x = r(x)

            x_penultimate = x
            x_ultimate = self.visual_backbone.transformer.resblocks[-1](x_penultimate)
            x = torch.cat([x_penultimate, x_ultimate], dim=0)

        x = self.ln_pre_out(x)
        x = self.linear_out(x)
        x = self.ln_post_out(x)

        x = x.permute(1, 0, 2).contiguous()

        return x

    def forward_xvlm(self, x: torch.Tensor):

        with torch.no_grad():
            x = self.visual_backbone.patch_embed(x)
            if self.visual_backbone.ape:
                x = x + self.visual_backbone.absolute_pos_embed
            x = self.visual_backbone.pos_drop(x)

            for layer in self.visual_backbone.layers[:-1]:
                x = layer(x)

            x_penultimate = x
            x_ultimate = self.visual_backbone.layers[-1](x)

            x_cls = self.visual_backbone.avgpool(self.visual_backbone.norm(x_ultimate).transpose(1, 2))  # B C 1
            x = torch.cat([x_cls.transpose(1, 2), x_ultimate], dim=1)

            x = torch.cat([x, x_penultimate], dim=1)

        x = self.ln_pre_out(x)
        x = self.linear_out(x)
        x = self.ln_post_out(x)

        return x

    def forward_instruct_blip(self, x: torch.Tensor):

        with torch.no_grad():
            x = self.visual_backbone.vision_model(x)["last_hidden_state"]
            prompt = einops.repeat(self.prompt, "1 L -> N L", N=x.shape[0])
            query_embeds = self.visual_backbone.query_tokens.expand(x.shape[0], -1, -1)
            prompt = self.visual_backbone.qformer.embeddings(input_ids=prompt,
                                                             query_embeds=query_embeds)
            x = self.visual_backbone.qformer.encoder(encoder_hidden_states=x,
                                                     hidden_states=prompt,
                                                     query_length=query_embeds.shape[1]).last_hidden_state
            x = x[:, :self.visual_backbone.query_tokens.shape[1], :]

        x = self.linear_out(x)
        x = self.ln_post_out(x)

        return x

    def train(self):
        self.decoder.train()
        if not self.instruct_blip:
            self.ln_pre_out.train()
        self.linear_out.train()
        self.ln_post_out.train()
        self.visual_backbone.eval()

    def eval(self):
        if not self.instruct_blip:
            self.ln_pre_out.eval()
        self.linear_out.eval()
        self.ln_post_out.eval()
        self.decoder.eval()
        self.visual_backbone.eval()


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)
