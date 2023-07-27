import json
import pandas as pd
raw_file = json.load(open("claude.json"))
max_tokens = 4096


def process(text):
    chat = []
    temp_chat = []
    if text["chat"][0]["role"] == "system" and text["chat"][0]["role"] == "system":
        system_prompt_text = text["chat"][0]["content"] + text["chat"][0]["role"]
        system_tokens = text["chat"][0]["token_count"] + text["chat"][1]["token_count"]
        text["chat"].pop(0)
        text["chat"].pop(1)
    elif text["chat"][0]["role"] == "system":
        system_prompt_text = text["chat"][0]["content"]
        system_tokens = text["chat"][0]["token_count"]
        text["chat"][0].pop()
    else:
        print("Nothing")
        system_prompt_text = "Roleplay between user and assitant"
        system_tokens = 10

    system_prompt = {
        "content": system_prompt_text,
        "role": "system",
        "token_count": system_tokens
    }
    remaining_tokens = max_tokens - system_tokens
    tokens_count = 0
    for convo in text["chat"]:        
        tokens_count += convo["token_count"]
        if tokens_count <= remaining_tokens:
            temp_chat.append(convo)
        if tokens_count > remaining_tokens:
            temp_chat.insert(0, system_prompt )
            chat.append(temp_chat)
            temp_chat = [convo]
            tokens_count = convo["token_count"]

    return chat

# Apply process to each line in raw_file and concatenate the resulting lists
processed_data = [item for sublist in [process(line) for line in raw_file] for item in sublist]

# Create a pandas DataFrame with a single column named "text"
df = pd.DataFrame({"text": processed_data})

# Save the DataFrame as a Parquet file
df.to_parquet("cleaned.parquet", index=False)