import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my_api_key",
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=836,
    temperature=1,
    system="Hello Claude this is Aman. I'm building this system to compress text with small ~7b param language models. For text x, you're gonna produce a compressed version such that P_{LM}(x | x') is maximized while minimizing the length of x'. You can use any prompting strategies you want. The user will give you the text x and you will respond with the compressed version. Note that the compressed version must be smaller than the input.\n\nYou must delimit your answer with the tags <c> and </c>. Note that newlines count toward the length! ",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hey, are you there?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Yes, I'm here. "
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Ok great. "
                }
            ]
        }
    ]
)
print(message.content)