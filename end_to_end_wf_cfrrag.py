######################################################################################################## 
# This script intends to using RAG to auto generate recommended actions based on an LLM question input

# Step 1: Parse the 21 CFR 803 into sections and subsections using regex using the Sectionalize class
from aiswre.preprocess.sectionalize import Sectionalize

parser = Sectionalize()

text = parser.get_pdf_text("./aiswre/data/Medical-Device-Reporting-for-Manufacturers--Guidance.pdf")

# Step 2: Construct a dataframe where each row contains the information needed
# to create a LCEL chain using the PromptFrame class

# Step 3: Run the chains asynchronously using the ParallelPromptRunner class

# Step 4: Run evaluation metrics