import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
from misc import pil_to_batched_tensor, colorize

def generate_dept_map(input_img, output_img):

    # ✅ Load MiDaS backbone (used in ZoeDepth)
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)

    # ✅ Load ZoeDepth model
    repo = "isl-org/ZoeDepth"
    # model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True) # for indoor only
    model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True) # for both indoor and outdoor

    # ✅ Set device (CPU/GPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    # ✅ Load Image for Depth Estimation
    image = Image.open(input_img).convert("RGB")

    # ✅ Predict Depth (as NumPy array)
    depth_numpy = zoe.infer_pil(image)  # Depth as NumPy array
    print(depth_numpy)
    print(f"Shape: {depth_numpy.shape}")
    print(f"Data Type: {depth_numpy.dtype}")
    print(f"Min: {np.min(depth_numpy)}")
    print(f"Max: {np.max(depth_numpy)}")
    print(f"Mean: {np.mean(depth_numpy)}")
    print(f"Median: {np.median(depth_numpy)}")
    print(f"Standard Deviation: {np.std(depth_numpy)}")
    print(f"25th Percentile: {np.percentile(depth_numpy, 25)}")
    print(f"75th Percentile: {np.percentile(depth_numpy, 75)}")
    # ✅ Convert depth to a colorized depth map (for visualization)
    depth_colorized = colorize(depth_numpy)

    # ✅ Save the Depth Map
    depth_pil = Image.fromarray(depth_colorized)
    depth_pil.save(output_img)

    print("✅ Depth Map Saved as 'depth_output.png' 🎉")

generate_dept_map("view.jpg", "output.jpg")    