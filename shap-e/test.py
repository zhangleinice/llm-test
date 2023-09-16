import torch
from diffusers import ShapEPipeline
import matplotlib.pyplot as plt

model_id = "openai/shap-e"
pipe = ShapEPipeline.from_pretrained(model_id).to("cuda")

guidance_scale = 15.0

prompt = "a beautiful girl"

image = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    # size=256,
).images[0][0]

print('image\n', image)

plt.imshow(image)
plt.axis('off')  # 不显示坐标轴

# 保存图片 .gif保存不了
plt.savefig("girl_3d.png")  # 保存图片


