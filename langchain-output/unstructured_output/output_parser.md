In **LangChain**, **Output Parsers** are components used to **process and structure the raw text output** returned by a language model (like GPT).

They help you **convert unstructured LLM responses** (plain text) into **structured data** that your program can easily use — such as JSON, lists, Python objects, or specific formats.

---

### 🧠 Why They’re Needed

When you ask an LLM for something like:

> “Give me 3 facts about AI.”

The model might reply:

```
1. AI stands for Artificial Intelligence.
2. It enables machines to mimic human intelligence.
3. It's used in healthcare, finance, and robotics.
```

But your app might need this as a structured Python list:

```python
["AI stands for Artificial Intelligence.",
 "It enables machines to mimic human intelligence.",
 "It's used in healthcare, finance, and robotics."]
```

That’s what an **Output Parser** does.

---

### ⚙️ Common Types of Output Parsers

1. **`StrOutputParser`**

   * Returns the raw string output (default).
   * Example: `parser = StrOutputParser()`

2. **`CommaSeparatedListOutputParser`**

   * Converts comma-separated text into a Python list.
   * Example: `"apple, banana, mango"` → `["apple", "banana", "mango"]`

3. **`PydanticOutputParser`**

   * Converts model output into a **Pydantic model** (for strict type validation).
   * Example:

     ```python
     class Person(BaseModel):
         name: str
         age: int
     parser = PydanticOutputParser(pydantic_object=Person)
     ```

4. **`StructuredOutputParser`**

   * Enforces a specific structure defined by a schema.
   * Often used with **`ResponseSchema`** objects.

5. **`JSONOutputParser` / `JsonOutputParser`**

   * Parses valid JSON text into a Python dictionary.

---

### 💡 Example

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

parser = CommaSeparatedListOutputParser()
prompt = PromptTemplate(
    template="List 5 programming languages, separated by commas.",
    input_variables=[]
)

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke(prompt.format())
parsed_output = parser.parse(response.content)
print(parsed_output)
```

**Output:**

```python
['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

---

### 🧩 In short:

**Output Parsers = Convert messy LLM text → clean structured data.**

They’re essential when you want to **reliably extract information** from LLMs and use it programmatically (e.g., in APIs, databases, or downstream logic).
