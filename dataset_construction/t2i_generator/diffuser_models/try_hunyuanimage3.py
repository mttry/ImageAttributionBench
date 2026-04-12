from transformers import AutoModelForCausalLM

# Load the model
model_id = "./HunyuanImage-3"
# Currently we can not load the model using HF model_id `tencent/HunyuanImage-3.0` directly 
# due to the dot in the name.

kwargs = dict(
    attn_implementation="sdpa",     # Use "flash_attention_2" if FlashAttention is installed
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",   # Use "flashinfer" if FlashInfer is installed
)

model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

# generate the image
prompt = "A brown and white dog is running on the grass"
image = model.generate_image(prompt=prompt, stream=True)
image.save("image.png")
