import torch 
import torch.nn
from pathlib import Path
import os
FILE = Path(__file__).resolve() #current directory
ROOT = FILE.parents[0]  #root directory
model_dir= os.path.join(ROOT, 'model.pt')
onnx_save_dir=os.path.join(ROOT, 'model.onnx')
# load model
model=torch.load(model_dir)
model.eval()
model = model.cuda()
input_name=['input']
output_name=['output']
#create a dummy input tensor
dummy_input=torch.randn(1,1,28,28)
dummy_input=dummy_input.cuda()
torch.onnx.export(model,dummy_input,onnx_save_dir,input_names=input_name,output_names=output_name, verbose=True)