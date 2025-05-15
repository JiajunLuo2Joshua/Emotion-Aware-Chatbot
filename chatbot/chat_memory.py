import openai

class ChatMemory:
    def __init__(self, system_prompt, max_tokens=3000):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _estimate_token_count(self):
        return sum(len(m["content"]) // 4 for m in self.messages)

    def add_user_input(self, text):
        self.messages.append({"role": "user", "content": text})
        self._trim_if_needed()

    def add_assistant_response(self, text):
        self.messages.append({"role": "assistant", "content": text})
        self._trim_if_needed()

    def _trim_if_needed(self):
        if self._estimate_token_count() > self.max_tokens:
            print("⚠️ Token count too high. Summarizing conversation history...")

            # Request GPT to summarize context
            summary_prompt = (
                "Please summarize the following assistant-user conversation in 1-2 sentences "
                "so I can remember the emotional context and main topic. Just summarize clearly, without extra notes:\n\n"
            )
            content = ""
            for m in self.messages[1:]:  # Skip system prompt
                prefix = "User: " if m["role"] == "user" else "Assistant: "
                content += f"{prefix}{m['content']}\n"

            try:
                summary_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You summarize emotional chatbot dialogue."},
                        {"role": "user", "content": summary_prompt + content}
                    ]
                )
                summary_text = summary_response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ Failed to summarize: {e}")
                summary_text = "[Summary unavailable due to error]"

            # Replace full history with summary
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "assistant", "content": summary_text}
            ]

    def get_messages(self):
        return self.messages
