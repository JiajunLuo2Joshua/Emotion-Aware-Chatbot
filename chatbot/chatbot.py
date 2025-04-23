import os
import openai

# ✅ Load OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    # Optional: You can hardcode the key here for testing (not recommended for production)
    # api_key = "sk-your-real-key"
    raise ValueError("❌ Please set the OPENAI_API_KEY environment variable.")

# ✅ Create OpenAI client
client = openai.OpenAI(api_key=api_key)

# ✅ Function to send a prompt to ChatGPT and get a response
def chatgpt_query(prompt, model="gpt-4o", temperature=0.5, max_tokens=150):
    """
    Sends a prompt to the ChatGPT API and returns the response.
    
    Args:
        prompt (str): The user prompt or message.
        model (str): Model name (e.g., "gpt-4o", "gpt-3.5-turbo").
        temperature (float): Randomness in response (0.0 = deterministic).
        max_tokens (int): Max number of tokens in the response.

    Returns:
        str: ChatGPT's reply.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Error communicating with ChatGPT API:", e)
        return None

# ✅ Command-line chatbot loop
if __name__ == "__main__":
    print("🤖 Welcome to the ChatGPT CLI chatbot! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = chatgpt_query(user_input)
        print("ChatGPT:", response)
