conda create -n sdgpa python=3.12

conda activate sdgpa

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install diffusers accelerate safetensors transformers

pip install matplotlib

pip install opencv-python 

pip install tyro 

pip install ipykernel