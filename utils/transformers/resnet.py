import torch


def resnet50():
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    
def resnet101():
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

def resnet152():
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)


if __name__ == '__main__':

    
    model = resnet152()
    print(model.fc.in_features)
    #print(checkpoint['patch_embed.proj.weight'])
    #print('*'*20)
    #model.load_state_dict(checkpoint)
    #model.load_weights(checkpoint,['*'])
    img = torch.randn((1,3,224,224))
    op = model(img)
    print(op.shape)