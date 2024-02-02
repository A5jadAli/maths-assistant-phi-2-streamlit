import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

# Load the model and tokenizer
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                             torch_dtype=torch.float32,
                                             device_map='auto',
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.2
)

# Ensure the tokenizer's pad token is set to match the EOS token
pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

# Initialize huggingface pipeline
local_llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template for the chatbot
template = """Act as a Math Assistant for students in grades 1 to 7, you will help solve math problem by guiding through the following steps:
1. Understand the Problem: Make sure you understand what the problem is asking.
2. Break it Down: Break the problem into smaller, more manageable parts.
3. Strategies to Solve: Think about different strategies that could be used to solve the problem.
4. Solve the Problem: Solve each part of the problem step-by-step.
5. Review: After solving, you'll review the solution to make sure it makes sense.

### Math Problem:
{instruction}

### Answer:
"""

# Create the Prompttemplate and LLMChain
prompt_template = PromptTemplate(template=template, 
                                 input_variables=["instruction"])
llm_chain = LLMChain(
    prompt=prompt_template,
    llm=local_llm
)

# Streamlit UI for Math Assistant
st.title("Math Assistant for Grades 1 to 7")

# User input for math problems
user_math_problem = st.text_area("Enter your math problem:", "", help="Write down the math problem you need help with.")

if st.button("Solve Problem"):
    # Check if the input is not empty
    if user_math_problem.strip():
        # Process the input
        response = llm_chain.run({"instruction": user_math_problem})
        # Display the structured guidance and solution
        st.write(response.get("output", ""))
    else:
        st.warning("Please enter a math problem before clicking 'Solve Math Problem'.")
