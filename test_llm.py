import os

# Prevent tests from making external LLM network calls by unsetting
# Groq/XAI keys if present in the environment.
os.environ["GROQ_API_KEY"] = ""
os.environ["XAI_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = ""

from jarvis import llm_reason

try:
    res = llm_reason("Say hello", timeout=20)
    with open("test_llm_output.txt", "w", encoding="utf-8") as f:
        f.write(str(res))
    print("Wrote output to test_llm_output.txt")
except Exception as e:
    with open("test_llm_output.txt", "w", encoding="utf-8") as f:
        f.write(f"EXCEPTION: {type(e)}: {e}")
    print("Wrote exception to test_llm_output.txt")
