
import PIL
from torchvision.transforms import ToTensor, Resize, CenterCrop

P = PIL.Image.open('pikachu3.jpg').convert('RGB')
P = P.resize((256, 256))
k =ToTensor()(P)
image = k.unsqueeze(0)
loc_pred, cls_pred = model(image.cuda())