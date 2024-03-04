from transformers import AutoTokenizer, MixtralConfig, FlaxMixtralModel

def main():

    config = MixtralConfig.from_pretrained("hf-internal-testing/Mixtral-tiny")
    
    # for i in range(20):
    #     i += 1
    #     print(i)
    # config.max_position_embeddings = 4092

    model = FlaxMixtralModel.from_pretrained("hf-internal-testing/Mixtral-tiny", config=config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="jax")

    x = model(**inputs)
    # print(x.last_hidden_state)

if __name__ == "__main__":

    main()