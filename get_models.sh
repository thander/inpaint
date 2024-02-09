wget https://civitai.com/api/download/models/130072 -O /workspace/models/realisticVisionV51.safetensors;
wget https://civitai.com/api/download/models/132760 -O /workspace/models/absolutereality.safetensors;
wget https://huggingface.co/CiaraRowles/TemporalDiff/resolve/main/temporaldiff-v1-animatediff.ckpt -O /workspace/temporaldiff-v1-animatediff.ckpt


wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin -O /workspace/cnmodels/ip-adapter-plus-face_sd15.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth -O /workspace/cnmodels/control_v11f1e_sd15_tile.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.yaml -O /workspace/cnmodels/control_v11f1e_sd15_tile.yaml
wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15_plus.pth -O /workspace/cnmodels/ip-adapter_sd15_plus.pth





wget https://civitai.com/api/download/models/130090 -O models/realisticVisionV51inpaint.safetensors;
git clone https://github.com/continue-revolution/sd-webui-segment-anything extensions/sd-webui-segment-anything
git clone https://github.com/w-e-w/sd-webui-nudenet-nsfw-censor extensions/sd-webui-nudenet-nsfw-censor
git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet


wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O  /workspace/extensions/sd-webui-segment-anything/models/sam/sam_vit_h_4b8939.pth
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -O /workspace/extensions/sd-webui-segment-anything/models/grounding-dino/groundingdino_swint_ogc.pth

# git clone https://github.com/IDEA-Research/GroundingDINO.git
# cd GroundingDINO
# pip install -e .

