from transformers import AutoTokenizer, FlaxMixtralModel

def main():

    model = FlaxMixtralModel.from_pretrained("hf-internal-testing/Mixtral-tiny")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="jax")

    x = model(**inputs)
    print(x.last_hidden_states)


if __name__ == "__main__":

    main()