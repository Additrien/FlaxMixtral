import numpy as np
import time
from jax import jit
from transformers import AutoTokenizer, FlaxMixtralForCausalLM, MixtralConfig

def jax_forward(model, inputs):
    outputs = model(**inputs)

if __name__ == "__main__":

    model_path = "mistralai/Mixtral-8x7B-v0.1"

    config = MixtralConfig.from_pretrained(model_path)
    config.max_position_embeddings = 32768
    model = FlaxMixtralForCausalLM.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="jax")

    forward = jit(model)
    forward(**inputs)

    loops = 100

    seq = []
    for i in range(loops):
        print(i)
        time1 = time.time()
        jax_forward(model, inputs)
        time2 = time.time()
        seq.append(time2-time1)
    
    print(f'{round(np.mean(seq)*100, 2)} ms +/- {round(np.std(seq)*100, 2)} ms per loop on {loops} loops')
