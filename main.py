from sam_mask import generate_mask
from inpaint import inpaint

window_img = "images/window.jpg"
reference_img = "images/reference.jpg"
mask_img = "images/mask.png"

prompt = (
    "Вертикальные тканевые шторы, ровные, тканевые, аккуратно расположены"
)

generate_mask(window_img, output_mask=mask_img)
inpaint(window_img, reference_img, mask_img, prompt)
