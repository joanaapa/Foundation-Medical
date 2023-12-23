import os
import math
import types
from torch import nn
from .bases import *
from utils.transformers.focal_dw import PatchEmbed as PE_Focal
from utils.transformers.dinov2_utils import new_forward_features
from utils.transformers import *
from torch.cuda.amp import autocast
from timm.models.vision_transformer import PatchEmbed as PE_Timm


class Identity(nn.Module):
    """An identity function."""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
  
class Classifier(BaseModel):
    """A wrapper class that provides different CNN backbones.
    
    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)

        if 'sam' in self.backbone_type:
            if self.pretrained:
                cpt = download_weights(backbone_type = self.backbone_type, patch_size = 16, pretrained_type = 'sam')
            self.backbone = transformers.__dict__[self.backbone_type](self.img_size, cpt)
            self.fc = nn.Linear(self.backbone.embed_dim, self.n_classes)

            if self.img_channels != 3:
                pretrained_weight = self.backbone.patch_embed.proj.weight.data
                if self.backbone.patch_embed.proj.bias is not None:
                    pretrained_bias = self.backbone.patch_embed.proj.bias.data
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :self.img_channels]
                
                self.backbone.patch_embed = transformers.sam.PatchEmbed(in_chans=self.img_channels)
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias 
            self.blocks = self.backbone.blocks
        
        elif 'dinov2' in self.backbone_type:
            self.backbone = transformers.__dict__[self.backbone_type]()
            if self.backbone_type == 'dinov2_vits14':
                self.fc = nn.Linear(384, self.n_classes)
            elif self.backbone_type == 'dinov2_vitl14':
                self.fc = nn.Linear(1024, self.n_classes)
            elif self.backbone_type == 'dinov2_vitg14':
                self.fc = nn.Linear(1536, self.n_classes)
            else:
                self.fc = nn.Linear(768, self.n_classes)
            if self.img_channels != 3:
                patch_embed_attrs = ["patch_size", "embed_dim"]
                patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}
                pretrained_weight = self.backbone.patch_embed.proj.weight.data
                if self.backbone.patch_embed.proj.bias is not None:
                    pretrained_bias = self.backbone.patch_embed.proj.bias.data
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :self.img_channels]
                
                self.backbone.patch_embed = transformers.dinov2_utils.PatchEmbed(in_chans=self.img_channels, **patch_defs)
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias 
            if self.partial.drop_cutoff != []:
                self.backbone.forward_features = types.MethodType(new_forward_features, self.backbone) 
            self.blocks = self.backbone.blocks
        
        elif 'focal' in self.backbone_type:
            self.backbone = transformers.D2FocalNet(self.img_size)
            if self.pretrained:
                cpt = download_weights(backbone_type = self.backbone_type, patch_size = 16, pretrained_type = 'seem')
                processed_dict = {'.'.join(k.split('.')[1:]): v for k, v in cpt.items() if k.startswith('backbone')}
                self.backbone.load_state_dict(processed_dict, strict=True)
            self.fc = nn.Linear(768, self.n_classes)
            if self.img_channels != 3:
                patch_embed_attrs = ["patch_size","embed_dim"]
                patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}
                pretrained_weight = self.backbone.patch_embed.proj.weight.data
                if self.backbone.patch_embed.proj.bias is not None:
                    pretrained_bias = self.backbone.patch_embed.proj.bias.data
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :self.img_channels]
                
                self.backbone.patch_embed = PE_Focal(in_chans=self.img_channels,is_stem=True,**patch_defs)
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias 
            self.blocks = self.backbone.layers # For focal we take each "block" as the different resolution, not the 'FocalModulationBlock'

        elif 'clip' in self.backbone_type:
            self.backbone = transformers.__dict__[self.backbone_type](self.img_size)
            self.fc = nn.Linear(512, self.n_classes)
            if self.img_channels != 3:
                patch_embed_attrs = ["kernel_size","stride"]
                patch_defs = {attr: getattr(self.backbone.conv1, attr) for attr in patch_embed_attrs}
                pretrained_weight = self.backbone.conv1.weight.data
                if self.backbone.conv1.bias is not None:
                    pretrained_bias = self.backbone.conv1.bias.data
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :self.img_channels]
                
                self.backbone.conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=768, bias=False ,**patch_defs)
                self.backbone.conv1.weight.data = pretrained_weight 
                if self.backbone.conv1.bias is not None:
                    self.backbone.conv1.bias.data = pretrained_bias
            self.blocks = self.backbone.transformer.resblocks

        elif 'resnet' in self.backbone_type:
            self.backbone = transformers.__dict__[self.backbone_type]()
            in_features = self.backbone.fc.in_features
            self.backbone.fc = Identity()
            self.fc = nn.Linear(in_features, self.n_classes)
            if self.img_channels != 3:
                conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                            'padding', 'dilation', "groups", "bias", "padding_mode"]
                conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

                pretrained_weight = self.backbone.conv1.weight.data
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :self.img_channels]

                self.backbone.conv1 = nn.Conv2d(self.img_channels, **conv1_defs)
                self.backbone.conv1.weight.data = pretrained_weight
            
        elif 'blip' in self.backbone_type:
            self.backbone = transformers.__dict__[self.backbone_type]()
            if self.pretrained:
                cpt = download_weights(backbone_type = self.backbone_type, patch_size = 16, pretrained_type = 'blip')
                processed_dict = {'.'.join(k.split('.')[1:]): v for k, v in cpt["model"].items() if k.startswith('visual_encoder')}
                self.backbone.load_state_dict(processed_dict,strict=True)
            self.fc = nn.Linear(768, self.n_classes)
            if self.img_channels != 3:
                pretrained_weight = self.backbone.patch_embed.proj.weight.data
                if self.backbone.patch_embed.proj.bias is not None:
                    pretrained_bias = self.backbone.patch_embed.proj.bias.data
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :self.img_channels]
                
                self.backbone.patch_embed = PE_Timm(img_size=224, in_chans=self.img_channels, embed_dim=768, patch_size=16)
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias 
            self.blocks = self.backbone.blocks
        else:
        
            if hasattr(transformers, self.backbone_type):  
                self.backbone = transformers.__dict__[self.backbone_type](**self.transformers_params, 
                                                                        pretrained=self.pretrained)
                fc_in_channels = self.backbone.num_features
                if hasattr(self.transformers_params, "trim_cutoff"):
                    self.trim_transformer()

            elif hasattr(cnn_models, self.backbone_type):
                self.backbone = cnn_models.__dict__[self.backbone_type](pretrained=self.pretrained, **incwargs)
                # loading non-standard weights
                pretrained_type = self.cnn_params.pretrained_type if hasattr(self, "cnn_params") else "supervised"
                if self.pretrained and pretrained_type != "supervised":
                    pre_cpt = download_cnn_weights(self.backbone_type, pretrained_type)
                    missed_keys = self.backbone.load_state_dict(pre_cpt, strict=False)
                    missing_head = set(missed_keys.missing_keys) == set(['fc.weight', 'fc.bias'])
                    unexpected_keys = missed_keys.unexpected_keys == []
                    is_ok = missing_head and unexpected_keys
                    if not is_ok:
                        raise ValueError(f"Found unexpected keys or keys are missing: {missed_keys}")
                    print_ddp(f"\033[96m Using pretrained type: {pretrained_type}\033[0m")
                fc_in_channels = self.backbone.fc.in_features
            else:
                raise NotImplementedError                
            self.backbone.fc = Identity()  # removing the fc layer from the backbone (which is manually added below)

            # modify stem and last layer
            self.fc = nn.Linear(fc_in_channels, self.n_classes)
            self.modify_first_layer(self.img_channels, self.pretrained)            
        
        if self.freeze_backbone:
            print_ddp(f"\033[96m Freezing backbone \033[0m")
            self.freeze_submodel(self.backbone)
            if hasattr(self, 'partial'):
                if self.partial.unfreeze_cutoff >= 0:
                    self.partial_unfreeze()
          

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            if self.freeze_backbone:
                self.backbone.eval()
                
            if isinstance(x, list) and hasattr(cnn_models, self.backbone_type):
                idx_crops = torch.cumsum(torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in x]),
                    return_counts=True,
                )[1], 0)
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx             
            else:
                if 'dinov2' in self.backbone_type:
                    if self.partial.drop_cutoff: # If a drop_cuttoff is specified
                        features = self.backbone.forward_features(x, return_block=self.partial.drop_cutoff)
                    else:
                        features = self.backbone.forward(x,is_training=True)
                    x_cls = self.backbone.head(features['x_norm_clstoken'])
                    x_emb = features['x_prenorm']
                    if return_embedding:
                        return self.fc(x_cls), x_cls
                    else:
                        if self.linear:
                            return self.fc(x_cls)
                        else:
                            return x_emb

                if 'sam' in self.backbone_type:
                    x_emb = self.backbone(x, return_block=self.partial.drop_cutoff)
                    x_cls = torch.mean(x_emb , dim = (1,2))
                    if return_embedding:
                        return self.fc(x_cls), x_cls
                    else:
                        if self.linear:
                            return self.fc(x_cls)
                        else:
                            return x_emb
                        
                
                if 'focal' in self.backbone_type:
                    out_layer = self.partial.drop_cutoff[0]+2 if self.partial.drop_cutoff else 5 # Layer 0 corresponds to resolution 2
                    x_emb = self.backbone(x)[f'res{out_layer}'].permute(0,2,3,1)
                    x_cls = torch.mean(x_emb , dim = (1,2))
                   
                    if return_embedding:
                        return self.fc(x_cls), x_cls
                    else:
                        if self.linear:
                            return self.fc(x_cls)
                        else:
                            return x_emb
                
                if 'clip' in self.backbone_type:
                    embed = self.backbone(x, return_block=self.partial.drop_cutoff)
                    x_cls, x_emb = embed,embed 
                    if return_embedding:
                        return self.fc(x_cls).float(), x_cls.float()
                    else:
                        if self.linear:
                            return self.fc(x_cls)
                        else:
                            return x_emb

                if 'blip' in self.backbone_type:
                    x_emb,x_cls = self.backbone(x, return_block=self.partial.drop_cutoff)
                    
                    if return_embedding:
                        return self.fc(x_cls), x_cls
                    else:
                        if self.linear:
                            return self.fc(x_cls)
                        else:
                            return x_emb
                        
            x_emb = self.backbone(x)
            x = self.fc(x_emb)
            
            if return_embedding:   
                return x, x_emb
            else:
                return x
            

    def partial_unfreeze(self):
        """Unfreeze the top blocks, including the unfreeze_cutoff block"""
        n_blocks = len(list(self.blocks))
        print_ddp(f"\033[96m Unfreezing backbone blocks {self.partial.unfreeze_cutoff} to {n_blocks-1}\033[0m")
        for i in range(self.partial.unfreeze_cutoff, n_blocks):
            self.unfreeze_submodel(self.blocks[i])

    def trim_transformer(self):
        """Remove the first blocks of the model, keep the rest, including the trim_cutoff"""
        cutoff = self.transformers_params.trim_cutoff
        if cutoff >=0:
            self.backbone.blocks = nn.Sequential(*list(self.backbone.blocks.children())[cutoff:])
        
    def modify_first_layer(self, img_channels, pretrained):
        backbone_type = self.backbone.__class__.__name__
        if img_channels == 3:
            return

        if backbone_type == 'ResNet':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight 
                
        elif backbone_type == 'Inception3':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.Conv2d_1a_3x3.conv, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.Conv2d_1a_3x3.conv.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.Conv2d_1a_3x3.conv.weight.data = pretrained_weight                 
                
        elif backbone_type == 'VisionTransformer':
            patch_embed_attrs = ["img_size", "patch_size", "embed_dim"]
            patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
            self.backbone.patch_embed = transformers.deit.PatchEmbed(in_chans=img_channels, **patch_defs)
            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias           
                    
        elif backbone_type == 'SwinTransformer':
            patch_embed_attrs = ["img_size", "patch_size", "embed_dim", "norm_layer"]
            patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data
            if self.backbone.patch_embed.norm is not None:
                norm_weight = self.backbone.patch_embed.norm.weight.data                
                norm_bias = self.backbone.patch_embed.norm.bias.data                
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
            self.backbone.patch_embed = transformers.swin.PatchEmbed(in_chans=img_channels, **patch_defs)
            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias      
                if self.backbone.patch_embed.norm is not None:
                    if self.backbone.patch_embed.norm.weight is not None:
                        self.backbone.patch_embed.norm.weight.data = norm_weight
                    if self.backbone.patch_embed.norm.bias is not None:
                        self.backbone.patch_embed.norm.bias.data = norm_bias
                        
        elif backbone_type == 'FocalTransformer':
            
            patch_embed_attrs = ["img_size", "patch_size", "embed_dim", "norm_layer",
                                 "use_conv_embed", "norm_layer", "use_pre_norm", "is_stem"]
            patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data
            if self.backbone.patch_embed.norm is not None:
                norm_weight = self.backbone.patch_embed.norm.weight.data                
                norm_bias = self.backbone.patch_embed.norm.bias.data 
            if self.backbone.patch_embed.pre_norm is not None:
                norm_weight = self.backbone.patch_embed.pre_norm.weight.data                
                norm_bias = self.backbone.patch_embed.pre_norm.bias.data                 
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
            self.backbone.patch_embed = transformers.focal.PatchEmbed(in_chans=img_channels, **patch_defs)
            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias      
                if self.backbone.patch_embed.norm is not None:
                    if self.backbone.patch_embed.norm.weight is not None:
                        self.backbone.patch_embed.norm.weight.data = norm_weight
                    if self.backbone.patch_embed.norm.bias is not None:
                        self.backbone.patch_embed.norm.bias.data = norm_bias 
                    if self.backbone.patch_embed.pre_norm.weight is not None:
                        self.backbone.patch_embed.pre_norm.weight.data = pre_norm_weight
                    if self.backbone.patch_embed.pre_norm.bias is not None:
                        self.backbone.patch_embed.pre_norm.bias.data = pre_norm_bias                         
        
        else:
            raise NotImplementedError("channel modification is not implemented for {}".format(backbone_type))


class Ymodel_SAM(BaseModel):
    def __init__(self, model_params, foundation_model, main_model):
        super().__init__()
        self.attr_from_dict(model_params)
        self.frozen_main = self.model_params.freeze_backbone
        self.frozen_foundation = self.foundation_params.freeze_backbone  
        self.embed_dim = main_model.backbone.embed_dim
        self.foundation_embed = foundation_model.backbone.embed_dim
        self.main_model = main_model
        self.foundation_model = foundation_model
                              
        self.linear_projector = nn.Sequential(nn.Linear(self.foundation_embed, self.embed_dim),
                                              nn.LayerNorm(self.embed_dim)
                                            )
        # send the wrapped model to the original model's GPU ID
        self.to(self.device_id)
                

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            self.freeze_foundation()
            found_embeds = self.foundation_model(x)
            linear_proj = self.linear_projector(found_embeds)

            batch_size = linear_proj.shape[0]
            embed_patches = linear_proj.shape[1]
            embed_dim = linear_proj.shape[3]
            linear_proj_reshaped = linear_proj.reshape(batch_size, embed_patches**2, embed_dim)
        
            if return_embedding:
                outputs, features = self.main_model(linear_proj_reshaped,return_embedding)
                return outputs, features
            else:
                outputs = self.main_model(linear_proj_reshaped,return_embedding)
                return outputs

    def freeze_check(self):
        self.freeze_foundation()
        self.freeze_main()
        
    def freeze_foundation(self):
        if self.frozen_foundation:
            self.foundation_model.eval()            
            self.freeze_submodel(self.foundation_model)
            
    def freeze_main(self):
        if self.frozen_main:
            self.main_model.eval() 
            self.freeze_submodel(self.main_model)
            pdb.set_trace()
            self.linear_projector.eval()            
            self.freeze_submodel(self.linear_projector)

class Ymodel_DINO(BaseModel):
    def __init__(self, model_params, foundation_model, main_model):
        super().__init__()
        self.attr_from_dict(model_params)
        self.frozen_main = self.model_params.freeze_backbone
        self.frozen_foundation = self.foundation_params.freeze_backbone  
        self.embed_dim = main_model.backbone.embed_dim
        self.foundation_embed = foundation_model.backbone.embed_dim
        self.main_model = main_model
        self.foundation_model = foundation_model
        self.linear_projector = nn.Sequential( nn.Linear(self.foundation_embed, self.embed_dim),
                                                nn.LayerNorm(self.embed_dim)
                                            )
       
        # send the wrapped model to the original model's GPU ID
        self.to(self.device_id)
                

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            self.freeze_foundation()
            
            found_embeds = self.foundation_model(x)[:,1:,:]
            linear_proj = self.linear_projector(found_embeds)

            if return_embedding:
                outputs, features = self.main_model(linear_proj,return_embedding)
                return outputs, features
            else:
                outputs = self.main_model(linear_proj,return_embedding)
                return outputs

    def freeze_check(self):
        self.freeze_foundation()
        self.freeze_main()
        
    def freeze_foundation(self):
        if self.frozen_foundation:
            self.foundation_model.eval()            
            self.freeze_submodel(self.foundation_model)
            
    def freeze_main(self):
        if self.frozen_main:
            self.main_model.eval() 
            self.freeze_submodel(self.main_model)
            pdb.set_trace()
            self.linear_projector.eval()            
            self.freeze_submodel(self.linear_projector)

class Ymodel_SEEM(BaseModel):
    def __init__(self, model_params, foundation_model, main_model):
        super().__init__()
        self.attr_from_dict(model_params)
        self.frozen_main = self.model_params.freeze_backbone
        self.frozen_foundation = self.foundation_params.freeze_backbone  
        self.embed_dim = main_model.backbone.embed_dim
        self.foundation_embed = foundation_model.backbone.embed_dim
        self.main_model = main_model
        self.foundation_model = foundation_model
        self.out_resolution = self.foundation_model.partial.drop_cutoff[0] if self.foundation_model.partial.drop_cutoff else 3
        self.linear_projector = nn.Sequential( nn.Linear(self.foundation_embed * 2**self.out_resolution,self.embed_dim),
                                                nn.LayerNorm(self.embed_dim)
                                            )
       
        # send the wrapped model to the original model's GPU ID
        self.to(self.device_id)
                

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            self.freeze_foundation()
            
            found_embeds = self.foundation_model(x)
            found_embeds = torch.flatten(found_embeds,1,2)
            linear_proj = self.linear_projector(found_embeds)

            if return_embedding:
                outputs, features = self.main_model(linear_proj,return_embedding)
                return outputs, features
            else:
                outputs = self.main_model(linear_proj,return_embedding)
                return outputs

    def freeze_check(self):
        self.freeze_foundation()
        self.freeze_main()
        
    def freeze_foundation(self):
        if self.frozen_foundation:
            self.foundation_model.eval()            
            self.freeze_submodel(self.foundation_model)
            
    def freeze_main(self):
        if self.frozen_main:
            self.main_model.eval() 
            self.freeze_submodel(self.main_model)
            pdb.set_trace()
            self.linear_projector.eval()            
            self.freeze_submodel(self.linear_projector)

class Ymodel_CLIP(BaseModel):
    def __init__(self, model_params, foundation_model, main_model):
        super().__init__()
        self.attr_from_dict(model_params)
        self.frozen_main = self.model_params.freeze_backbone
        self.frozen_foundation = self.foundation_params.freeze_backbone  
        self.embed_dim = main_model.backbone.embed_dim
        self.foundation_embed = 768
        self.main_model = main_model
        self.foundation_model = foundation_model
        self.linear_projector = nn.Sequential( nn.Linear(self.foundation_embed, self.embed_dim),
                                                nn.LayerNorm(self.embed_dim)
                                            )
       
        # send the wrapped model to the original model's GPU ID
        self.to(self.device_id)
                

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            self.freeze_foundation()
            found_embeds = self.foundation_model(x)
            #print(found_embeds.shape)
            linear_proj = self.linear_projector(found_embeds)

            if return_embedding:
                outputs, features = self.main_model(linear_proj,return_embedding)
                return outputs, features
            else:
                outputs = self.main_model(linear_proj,return_embedding)
                return outputs

    def freeze_check(self):
        self.freeze_foundation()
        self.freeze_main()
        
    def freeze_foundation(self):
        if self.frozen_foundation:
            self.foundation_model.eval()            
            self.freeze_submodel(self.foundation_model)
            
    def freeze_main(self):
        if self.frozen_main:
            self.main_model.eval() 
            self.freeze_submodel(self.main_model)
            pdb.set_trace()
            self.linear_projector.eval()            
            self.freeze_submodel(self.linear_projector)
