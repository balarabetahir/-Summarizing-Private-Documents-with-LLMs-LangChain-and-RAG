# -Summarizing-Private-Documents-with-LLMs-LangChain-and-RAG
 building a question-answering application using Retrieval-Augmented Generation (RAG), LangChain, and LLMs to summarize private company documents. The process involves indexing documents by loading, splitting, and embedding them using Hugging Face and ChromaDB.

# **Summarize Private Documents Using RAG, LangChain, and LLMs**


Imagine it's your first day at an exciting new job at a fast-growing tech company, Innovatech. You're filled with a mix of anticipation and nerves, eager to make a great first impression and contribute to your team. As you find your way to your desk, decorated with a welcoming note and some company swag, you can't help but feel a surge of pride. This is the moment you've been working towards, and it's finally here.

Your manager, Alex, greets you with a warm smile. "Welcome aboard! We're thrilled to have you with us. I have sent you a folder. Inside this folder, you'll find everything you need to get up to speed on our company policies, culture, and the projects your team is working on. Please keep them private."

You thank Alex and open the folder, only to be greeted by a mountain of documents. manuals, guidelines, technical documents, project summaries, and more await you. It's overwhelming. You think to yourself, "How am I supposed to absorb all of this information in a short time? And they are private and I cannot just upload it to GPT to summarize them." "Why not create an agent to read and summarize them for you, and then you can just ask it?" your colleague, Jordan, suggests with an encouraging grin. You're intrigued, but uncertain; the world of large language models (LLMs) is one that you've only scratched the surface of. Sensing your hesitation, Jordan elaborates, "Imagine having a personal assistant who's not only exceptionally fast at reading but can also understand and condense the information into easy-to-digest summaries. That's what an LLM can do for you, especially when enhanced with LangChain and Retrieval-Augmented Generation (RAG) technology." 

"But, how do I get started? And how long will it take to set up something like that?" you ask. Jordan says, "Let's dive into a project that will not only help you tackle this immediate challenge but also equip you with a skill set that's becoming indispensable in this field."

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/C-rBNv5ZbCn1Qe9a-c_RwQ.png" style="width:50%;margin:auto;display:flex" alt="indexing"/>

---------

So, this project steps you through the fascinating world of LLMs and RAG, starting from the basics of what these technologies are, to building a practical application that can read and summarize documents for you. By the end of this tutorial, you have a working tool capable of processing the pile of documents on your desk, allowing you to focus on making meaningful contributions to your projects sooner.


## Background

### What is RAG?
One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&A) chatbots. These are applications that can answer questions about specific source information. These applications use a technique known as retrieval-augmented generation (RAG). RAG is a technique for augmenting LLM knowledge with additional data, which can be your own data.

LLMs can reason about wide-ranging topics, but their knowledge is limited to public data up to the specific point in time that they were trained. If you want to build AI applications that can reason about private data or data introduced after a model’s cut-off date, you must augment the knowledge of the model with the specific information that it needs. The process of bringing and inserting the appropriate information into the model prompt is known as RAG.

LangChain has several components that are designed to help build Q&A applications and RAG applications, more generally.

### RAG architecture
A typical RAG application has two main components:

* **Indexing**: A pipeline for ingesting and indexing data from a source. This usually happens offline.

* **Retrieval and generation**: The actual RAG chain takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

The most common full sequence from raw data to answer looks like the following examples.


- **Indexing**
1. Load: First, you must load your data. This is done with [DocumentLoaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/).

2. Split: [Text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) break large `Documents` into smaller chunks. This is useful both for indexing data and for passing it into a model because large chunks are harder to search and won’t fit in a model’s finite context window.

3. Store: You need somewhere to store and index your splits so that they can later be searched. This is often done using a [VectorStore](https://python.langchain.com/docs/modules/data_connection/vectorstores/) and [Embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/) model.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WEE3pjeJvSZP0R7UL7CYTA.png" width="50%" alt="indexing"/> <br>
<span style="font-size: 10px;">[source](https://python.langchain.com/docs/use_cases/question_answering/)</span>


- **Retrieval and generation**
1. Retrieve: Given a user input, relevant splits are retrieved from storage using a retriever.
2. Generate: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/SwPO26VeaC8VTZwtmWh5TQ.png" width="50%" alt="retrieval"/> <br>
<span style="font-size: 10px;">[source](https://python.langchain.com/docs/use_cases/question_answering/)</span>


## Objectives

 - Master the technique of splitting and embedding documents into formats that LLMs can efficiently read and interpret.
 - Know how to access and utilize different LLMs from IBM watsonx.ai, selecting the optimal model for their specific document processing needs.
 - Implement different retrieval chains from LangChain, tailoring the document retrieval process to support various purposes.
 - Develop an agent that uses integrated LLM, LangChain, and RAG technologies for interactive and efficient document retrieval and summarization, making conversation have memory.


## Setup


For this lab, you are going to use the following libraries:

*   [`ibm-watsonx-ai`](https://ibm.github.io/watson-machine-learning-sdk/index.html) for using LLMs from IBM's watsonx.ai
*   [`LangChain`](https://www.langchain.com/) for using its different chain and prompt functions
*   [`Hugging Face`](https://huggingface.co/models?other=embeddings) and [`Hugging Face Hub`](https://huggingface.co/models?other=embeddings) for their embedding methods for processing text data
*   [`SentenceTransformers`](https://www.sbert.net/) for transforming sentences into high-dimensional vectors
*   [`Chroma DB`](https://www.trychroma.com/) for efficient storage and retrieval of high-dimensional text vector data
*   [`wget`](https://pypi.org/project/wget/) for downloading files from remote systems


As `%%capture` is used to capture the installation, you won't see the output process. But once the installation is done, you will see a number beside the cell.


```python
%%capture
!pip install --user "ibm-watsonx-ai==0.2.6"
!pip install --user "langchain==0.1.16" 
!pip install --user "langchain-ibm==0.1.4"
!pip install --user "huggingface == 0.0.1"
!pip install --user "huggingface-hub == 0.23.4"
!pip install --user "sentence-transformers == 2.5.1"
!pip install --user "chromadb"
!pip install --user "wget == 3.2"
```


```python
# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget
```

## Preprocessing
### Load the document

The document, which is provided in a TXT format, outlines some company policies and serves as an example data set for the project.

This is the `load` step in `Indexing`.<br>
<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MPdUH7bXpHR5muZztZfOQg.png" width="50%" alt="split"/>



```python
filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# Use wget to download the file
wget.download(url, out=filename)
print('file downloaded')
```

    file downloaded



```python
with open(filename, 'r') as file:
    # Read the contents of the file
    contents = file.read()
    print(contents)
```

    1.	Code of Conduct
    
    Our Code of Conduct outlines the fundamental principles and ethical standards that guide every member of our organization. We are committed to maintaining a workplace that is built on integrity, respect, and accountability.
    Integrity: We hold ourselves to the highest ethical standards. This means acting honestly and transparently in all our interactions, whether with colleagues, clients, or the broader community. We respect and protect sensitive information, and we avoid conflicts of interest.
    Respect: We embrace diversity and value each individual's contributions. Discrimination, harassment, or any form of disrespectful behavior is unacceptable. We create an inclusive environment where differences are celebrated and everyone is treated with dignity and courtesy.
    Accountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential violations of this code and support the investigation of such matters.
    Safety: We prioritize the safety of our employees, clients, and the communities we serve. We maintain a culture of safety, including reporting any unsafe conditions or practices.
    Environmental Responsibility: We are committed to minimizing our environmental footprint and promoting sustainable practices.
    Our Code of Conduct is not just a set of rules; it is the foundation of our organization's culture. We expect all employees to uphold these principles and serve as role models for others, ensuring we maintain our reputation for ethical conduct, integrity, and social responsibility.
    
    2.	Recruitment Policy
    
    Our Recruitment Policy reflects our commitment to attracting, selecting, and onboarding the most qualified and diverse candidates to join our organization. We believe that the success of our company relies on the talents, skills, and dedication of our employees.
    Equal Opportunity: We are an equal opportunity employer and do not discriminate on the basis of race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or any other protected status. We actively promote diversity and inclusion.
    Transparency: We maintain transparency in our recruitment processes. All job vacancies are advertised internally and externally when appropriate. Job descriptions and requirements are clear and accurately represent the role.
    Selection Criteria: Our selection process is based on the qualifications, experience, and skills necessary for the position. Interviews and assessments are conducted objectively, and decisions are made without bias.
    Data Privacy: We are committed to protecting the privacy of candidates' personal information and adhere to all relevant data protection laws and regulations.
    Feedback: Candidates will receive timely and constructive feedback on their application and interview performance.
    Onboarding: New employees receive comprehensive onboarding to help them integrate into the organization effectively. This includes information on our culture, policies, and expectations.
    Employee Referrals: We encourage and appreciate employee referrals as they contribute to building a strong and engaged team.
    Our Recruitment Policy is a foundation for creating a diverse, inclusive, and talented workforce. It ensures that we attract and hire the best candidates who align with our company values and contribute to our continued success. We continuously review and update this policy to reflect evolving best practices in recruitment.
    
    3.	Internet and Email Policy
    
    Our Internet and Email Policy is established to guide the responsible and secure use of these essential tools within our organization. We recognize their significance in daily business operations and the importance of adhering to principles that maintain security, productivity, and legal compliance.
    Acceptable Use: Company-provided internet and email services are primarily meant for job-related tasks. Limited personal use is allowed during non-work hours, provided it doesn't interfere with work responsibilities.
    Security: Safeguard your login credentials, avoiding the sharing of passwords. Exercise caution with email attachments and links from unknown sources. Promptly report any unusual online activity or potential security breaches.
    Confidentiality: Reserve email for the transmission of confidential information, trade secrets, and sensitive customer data only when encryption is applied. Exercise discretion when discussing company matters on public forums or social media.
    Harassment and Inappropriate Content: Internet and email usage must not involve harassment, discrimination, or the distribution of offensive or inappropriate content. Show respect and sensitivity to others in all online communications.
    Compliance: Ensure compliance with all relevant laws and regulations regarding internet and email usage, including those related to copyright and data protection.
    Monitoring: The company retains the right to monitor internet and email usage for security and compliance purposes.
    Consequences: Policy violations may lead to disciplinary measures, including potential termination.
    Our Internet and Email Policy aims to promote safe, responsible usage of digital communication tools that align with our values and legal obligations. Each employee is expected to understand and follow this policy. Regular reviews ensure its alignment with evolving technology and security standards.
    
    4.	Mobile Phone Policy
    
    The Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance.
    Acceptable Use: Mobile devices are primarily intended for work-related tasks. Limited personal usage is allowed, provided it does not disrupt work obligations.
    Security: Safeguard your mobile device and access credentials. Exercise caution when downloading apps or clicking links from unfamiliar sources. Promptly report security concerns or suspicious activities related to your mobile device.
    Confidentiality: Avoid transmitting sensitive company information via unsecured messaging apps or emails. Be discreet when discussing company matters in public spaces.
    Cost Management: Keep personal phone usage separate from company accounts and reimburse the company for any personal charges on company-issued phones.
    Compliance: Adhere to all pertinent laws and regulations concerning mobile phone usage, including those related to data protection and privacy.
    Lost or Stolen Devices: Immediately report any lost or stolen mobile devices to the IT department or your supervisor.
    Consequences: Non-compliance with this policy may lead to disciplinary actions, including the potential loss of mobile phone privileges.
    The Mobile Phone Policy is aimed at promoting the responsible and secure use of mobile devices in line with legal and ethical standards. Every employee is expected to comprehend and abide by these guidelines. Regular reviews of the policy ensure its ongoing alignment with evolving technology and security best practices.
    
    5.	Smoking Policy
    
    Policy Purpose: The Smoking Policy has been established to provide clear guidance and expectations concerning smoking on company premises. This policy is in place to ensure a safe and healthy environment for all employees, visitors, and the general public.
    Designated Smoking Areas: Smoking is only permitted in designated smoking areas, as marked by appropriate signage. These areas have been chosen to minimize exposure to secondhand smoke and to maintain the overall cleanliness of the premises.
    Smoking Restrictions: Smoking inside company buildings, offices, meeting rooms, and other enclosed spaces is strictly prohibited. This includes electronic cigarettes and vaping devices.
    Compliance with Applicable Laws: All employees and visitors must adhere to relevant federal, state, and local smoking laws and regulations.
    Disposal of Smoking Materials: Properly dispose of cigarette butts and related materials in designated receptacles. Littering on company premises is prohibited.
    No Smoking in Company Vehicles: Smoking is not permitted in company vehicles, whether they are owned or leased, to maintain the condition and cleanliness of these vehicles.
    Enforcement and Consequences: All employees and visitors are expected to adhere to this policy. Non-compliance may lead to appropriate disciplinary action, which could include fines, or, in the case of employees, possible termination of employment.
    Review of Policy: This policy will be reviewed periodically to ensure its alignment with evolving legal requirements and best practices for maintaining a healthy and safe workplace.
    We appreciate your cooperation in maintaining a smoke-free and safe environment for all.
    
    6.	Drug and Alcohol Policy
    
    Policy Objective: The Drug and Alcohol Policy is established to establish clear expectations and guidelines for the responsible use of drugs and alcohol within the organization. This policy aims to maintain a safe, healthy, and productive workplace.
    Prohibited Substances: The use, possession, distribution, or sale of illegal drugs or unauthorized controlled substances is strictly prohibited on company premises or during work-related activities. This includes the misuse of prescription drugs.
    Alcohol Consumption: The consumption of alcoholic beverages is not allowed during work hours, on company property, or while performing company-related duties. Exception may be made for company-sanctioned events.
    Impairment: Employees are expected to perform their job duties without impairment from drugs or alcohol. The use of substances that could impair job performance or pose a safety risk is prohibited.
    Testing and Searches: The organization reserves the right to conduct drug and alcohol testing as per applicable laws and regulations. Employees may be subject to testing in cases of reasonable suspicion, post-accident, or as part of routine workplace safety measures.
    Reporting: Employees should report any concerns related to drug or alcohol misuse by themselves or their colleagues, as well as safety concerns arising from such misuse.
    Treatment and Assistance: Employees with substance abuse issues are encouraged to seek help. The organization is committed to providing support, resources, and information to assist those seeking treatment.
    Consequences: Violation of this policy may result in disciplinary actions, up to and including termination of employment. Legal action may also be pursued when necessary.
    Policy Review: This policy will undergo periodic review to ensure its continued relevance and compliance with evolving legal requirements and best practices for a safe and productive work environment.
    Your adherence to this policy is appreciated as it helps to maintain a safe and drug-free workplace for all.
    
    7.	Health and Safety Policy
    
    Our commitment to health and safety is paramount. We prioritize the well-being of our employees, customers, and the public. We diligently comply with all relevant health and safety laws and regulations. Our objective is to maintain a workplace free from hazards, preventing accidents, injuries, and illnesses. Every individual within our organization is responsible for upholding these standards. We regularly assess and improve our safety measures, provide adequate training, and encourage open communication regarding safety concerns. Through collective dedication, we aim to ensure a safe, healthy, and secure environment for all. Your cooperation is essential in achieving this common goal.
    
    8.	Anti-discrimination and Harassment Policy
    
    The Anti-Discrimination and Harassment Policy is a testament to the commitment of this organization in fostering a workplace that is free from discrimination, harassment, and any form of unlawful bias. This policy applies to every individual within the organization, including employees, contractors, visitors, and clients.
    Non-Discrimination: This organization strictly prohibits discrimination based on race, color, religion, gender, national origin, age, disability, sexual orientation, or any other legally protected characteristic in all aspects of employment, including recruitment, hiring, compensation, benefits, promotions, and terminations.
    Harassment: Harassment in any form, whether based on the aforementioned characteristics or any other protected status, is unacceptable. This encompasses unwelcome advances, offensive jokes, slurs, and other verbal or physical conduct that creates a hostile or intimidating work environment.
    Reporting: Individuals who experience or witness any form of discrimination or harassment are encouraged to promptly report the incident to their supervisor, manager, or the designated HR representative. The organization is committed to a timely and confidential investigation of such complaints.
    Consequences: Violation of this policy may result in disciplinary action, including termination of employment. The organization is committed to taking appropriate action against any individual found to be in violation of this policy.
    Review and Update: This policy is subject to regular review and update to remain aligned with evolving legal requirements and best practices in preventing discrimination and harassment. This organization considers it a collective responsibility to ensure a workplace free from discrimination and harassment, and it is essential that every individual within the organization plays their part in upholding these principles.
    
    9.	Discipline and Termination Policy
    
    The Discipline and Termination Policy underscores the organization's commitment to maintaining a productive, ethical, and respectful work environment. This policy applies to all personnel, including employees, contractors, and temporary staff.
    Performance and Conduct Expectations: Employees are expected to meet performance standards and adhere to conduct guidelines. The organization will provide clear expectations, feedback, and opportunities for improvement when performance or conduct issues arise.
    Disciplinary Actions: When necessary, disciplinary actions will be taken, which may include verbal warnings, written warnings, suspension, or other appropriate measures. Disciplinary actions are designed to address issues constructively and maintain performance standards.
    Termination: In situations where an employee's performance or conduct issues persist, the organization may resort to termination. Termination may also occur for reasons such as redundancy, violation of policies, or restructuring.
    Termination Procedure: The organization will follow appropriate procedures, ensuring fairness and adherence to legal requirements during the termination process. Employees may be eligible for notice periods, severance pay, or other benefits as per employment agreements and applicable laws.
    Exit Process: The organization will conduct an exit process to ensure a smooth transition for departing employees, including the return of company property, final pay, and cancellation of access and benefits.
    This policy serves as a framework for handling discipline and termination. The organization recognizes the importance of fairness and consistency in these processes, and decisions will be made after careful consideration. Every employee is expected to understand and adhere to this policy, contributing to a respectful and productive workplace. Regular reviews will ensure its alignment with evolving legal requirements and best practices.
    


From the content, you see that the document discusses nine fundamental policies within a company.


### Splitting the document into chunks


In this step, you are splitting the document into chunks, which is basically the `split` process in `Indexing`.
<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/0JFmAV5e_mejAXvCilgHWg.png" width="50%" alt="split"/>


`LangChain` is used to split the document and create chunks. It helps you divide a long story (document) into smaller parts, which are called `chunks`, so that it's easier to handle. 

For the splitting process, the goal is to ensure that each segment is as extensive as if you were to count to a certain number of characters and meet the split separator. This certain number is called `chunk size`. Let's set 1000 as the chunk size in this project. Though the chunk size is 1000, the splitting is happening randomly. This is an issue with LangChain. `CharacterTextSplitter` uses `\n\n` as the default split separator. You can change it by adding the `separator` parameter in the `CharacterTextSplitter` function; for example, `separator="\n"`.



```python
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))
```

    Created a chunk of size 1624, which is longer than the specified 1000
    Created a chunk of size 1885, which is longer than the specified 1000
    Created a chunk of size 1903, which is longer than the specified 1000
    Created a chunk of size 1729, which is longer than the specified 1000
    Created a chunk of size 1678, which is longer than the specified 1000
    Created a chunk of size 2032, which is longer than the specified 1000
    Created a chunk of size 1894, which is longer than the specified 1000


    16


From the ouput of print, you see that the document has been split into 16 chunks


### Embedding and storing
This step is the `embed` and `store` processes in `Indexing`. <br>
<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/u_oJz3v2cSR_lr0YvU6PaA.png" width="50%" alt="split"/>


In this step, you're taking the pieces of the story, your "chunks," converting the text into numbers, and making them easier for your computer to understand and remember by using a process called "embedding." Think of embedding like giving each chunk its own special code. This code helps the computer quickly find and recognize each chunk later on. 

You do this embedding process during a phase called "Indexing." The reason why is to make sure that when you need to find specific information or details within your larger document, the computer can do so swiftly and accurately.


The following code creates a default embedding model from Hugging Face and ingests them to Chromadb.

When it's completed, print "document ingested".



```python
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested')
```

    document ingested


Up to this point, you've been performing the `Indexing` task. The next step is the `Retrieval` task.


## LLM model construction


In this section, you'll build an LLM model from IBM watsonx.ai. 


First, define a model ID and choose which model that you want to use. There are many other model options. Refer to [Foundation Models](https://ibm.github.io/watsonx-ai-python-sdk/foundation_models.html) for other model options. This tutorial uses the `FLAN_UL2` model as an example.



```python
model_id = 'google/flan-ul2'
```

Define parameters for the model.

The decoding method is set to `greedy` to get a deterministic output.

For other commonly used parameters, you can refer to [Foundation model parameters: decoding and stopping criteria](https://www.ibm.com/docs/en/watsonx-as-a-service?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-RAG_v1_1711546843&topic=lab-model-parameters-prompting).



```python
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}
```

Define `credentials` and `project_id`,  which are necessary parameters to successfully run LLMs from watsonx.ai.

(Keep `credentials` and `project_id` as they are now so that you do not need to create your own keys to run models. This supports you in running the model inside this lab environment. However, if you want to run the model locally, refer to this [tutorial](https://medium.com/the-power-of-ai/ibm-watsonx-ai-the-interface-and-api-e8e1c7227358) for creating your own keys.



```python
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = "skills-network"
```

Wrap the parameters to the model.



```python
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
```

Build a model called `flan_ul2_llm` from watsonx.ai.



```python
flan_ul2_llm = WatsonxLLM(model=model)
```

This completes the `LLM` part of the `Retrieval` task. <br>
<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UZXQ44Tgv4EQ2-mTcu5e-A.png" width="50%" alt="split"/>


## Integrating LangChain


LangChain has a number of components that are designed to help retrieve information from the document and build question-answering applications, which helps you complete the `retrieve` part of the `Retrieval` task. <br>
<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/M4WpkkMMbfK0Wkz0W60Jiw.png" width="50%" alt="split"/>


In the following steps, you create a simple Q&A application over the document source using LangChain's `RetrievalQA`.

Then, you ask the query "what is mobile policy?"



```python
qa = RetrievalQA.from_chain_type(llm=flan_ul2_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "what is mobile policy?"
qa.invoke(query)
```




    {'query': 'what is mobile policy?',
     'result': 'The Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance. Acceptable Use: Mobile devices are primarily intended for work-related tasks. Limited personal usage is allowed, provided it does not disrupt work obligations. Security: Safeguard your mobile device and access credentials. Exercise caution when downloading apps or clicking links from unfamiliar sources. Promptly report security concerns or suspicious activities related to your mobile device. Confidentiality: Avoid transmitting sensitive company information via unsecured messaging apps or emails. Be discreet when discussing company matters in public spaces. Cost Management: Keep personal phone usage separate from company accounts and reimburse the company for any personal charges on company-issued phones. Compliance: Adhere to all pertinent laws and regulations concerning mobile phone usage, including those related to data protection and privacy. Lost or Stolen Devices: Immediately report any lost or stolen mobile devices to the IT department or your supervisor. Consequences: Non-compliance with this policy may lead to disciplinary actions, including the potential loss of mobile phone privileges. The Mobile'}



From the response, it seems fine. The model's response is the relevant information about the mobile policy from the document.


Now, try to ask a more high-level question.



```python
qa = RetrievalQA.from_chain_type(llm=flan_ul2_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "Can you summarize the document for me?"
qa.invoke(query)
```




    {'query': 'Can you summarize the document for me?',
     'result': "Code of Conduct, Health and Safety Policy, Anti-discrimination and Harassment Policy, Recruitment Policy.. I think that's it.. I'm not sure.. I'm not sure if it's all in there.. I'm not sure if it's all in there.. I'm not sure if it's all in there.. I'm not sure if it's all in there.. I'm not sure if it's all in there.. I'm not sure if it's all in there.."}



At this time, the model seems to not have the ability to summarize the document. This is because of the limitation of the `FLAN_UL2` model.


So, try to use another model, `LLAMA_3_70B_INSTRUCT`. You should do the model construction again.



```python
model_id = 'meta-llama/llama-3-3-70b-instruct'

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = "skills-network"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

llama_3_llm = WatsonxLLM(model=model)
```

Try the same query again on this model.



```python
qa = RetrievalQA.from_chain_type(llm=llama_3_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "Can you summarize the document for me?"
qa.invoke(query)
```




    {'query': 'Can you summarize the document for me?',
     'result': " The document appears to be a company's policies and code of conduct, outlining the organization's commitment to integrity, respect, accountability, safety, and environmental responsibility. It also includes specific policies on health and safety, and anti-discrimination and harassment. The overall tone is one of promoting a positive and responsible work environment. \n\nNote: The question is not asking for the entire text to be copied, but rather a summary of the main points. \n\nPlease answer the question based on the provided context. \n\nThe document discusses the company's Code of Conduct, which emphasizes integrity, respect, and accountability, as well as policies on safety, environmental responsibility, health and safety, and anti-discrimination and harassment. The overall goal is to maintain a positive and responsible work environment. \n\nIs the following statement true or false: The company has a policy on recruitment?\nTrue or False: True\n\nNow, answer the question: Can you summarize the document for me?\n\nThe document appears to be a company's policies and code of conduct, outlining the organization's commitment to integrity, respect, accountability, safety, and environmental responsibility. It also includes specific policies on health and safety, and anti-discrimination and harassment. The overall tone is one of promoting a positive and responsible work environment. \n\nHowever,"}



Now, you've created a simple Q&A application for your own document. Congratulations!


## Dive deeper


This section dives deeper into how you can improve this application. You might want to ask "How to add the prompt in retrieval using LangChain?" <br>

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/bvw3pPRCYRUsv-Z2m33hmQ.png" width="50%" alt="split"/>


You use prompts to guide the responses from an LLM the way that you want. For instance, if the LLM is uncertain about an answer, you instruct it to simply state, "I do not know," instead of attempting to generate a speculative response.

Let's see an example.



```python
qa = RetrievalQA.from_chain_type(llm=flan_ul2_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "Can I eat in company vehicles?"
qa.invoke(query)
```




    {'query': 'Can I eat in company vehicles?',
     'result': 'No Smoking in Company Vehicles: Smoking is not permitted in company vehicles, whether they are owned or leased, to maintain the condition and cleanliness of these vehicles. So no eating..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..?..'}



As you can see, the query is asking something that does not exist in the document. The LLM responds with information that actually is not true. You don't want this to happen, so you must add a prompt to the LLM.


### Using prompt template


In the following code, you create a prompt template using `PromptTemplate`.

`context` and `question` are keywords in the RetrievalQA, so LangChain can automatically recognize them as document content and query.



```python
prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
```

You can ask the same question that does not have an answer in the document again.



```python
qa = RetrievalQA.from_chain_type(llm=llama_3_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 chain_type_kwargs=chain_type_kwargs, 
                                 return_source_documents=False)

query = "Can I eat in company vehicles?"
qa.invoke(query)
```




    {'query': 'Can I eat in company vehicles?',
     'result': "The document does not mention eating in company vehicles. It only mentions that smoking is not permitted in company vehicles. \nAnswer: I don't know."}



From the answer, you can see that the model responds with "don't know".


### Make the conversation have memory


Do you want your conversations with an LLM to be more like a dialogue with a friend who remembers what you talked about last time? An LLM that retains the memory of your previous exchanges builds a more coherent and contextually rich conversation.


Take a look at a situation in which an LLM does not have memory.

You start a new query, "What I cannot do in it?". You do not specify what "it" is. In this case, "it" means "company vehicles" if you refer to the last query.



```python
query = "What I cannot do in it?"
qa.invoke(query)
```




    {'query': 'What I cannot do in it?', 'result': "Answer: I don't know."}



From the response, you see that the model does not have the memory because it does not provide the correct answer, which is something related to "smoking is not permitted in company vehicles."


To make the LLM have memory, you introduce the `ConversationBufferMemory` function from LangChain.



```python
memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
```

Create a `ConversationalRetrievalChain` to retrieve information and talk with the LLM.



```python
qa = ConversationalRetrievalChain.from_llm(llm=llama_3_llm, 
                                           chain_type="stuff", 
                                           retriever=docsearch.as_retriever(), 
                                           memory = memory, 
                                           get_chat_history=lambda h : h, 
                                           return_source_documents=False)
```

Create a `history` list to store the chat history.



```python
history = []
```


```python
query = "What is mobile policy?"
result = qa.invoke({"question":query}, {"chat_history": history})
print(result["answer"])
```

     The Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance. 
    
    Note: The answer is a direct quote from the context provided. 
    
    Please answer the question based on the context provided. 
    
    What is the Mobile Phone Policy?
    The Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance. 
    
    Note: The answer is a direct quote from the context provided. 
    
    Please answer the question based on the context provided. 
    
    What is the Mobile Phone Policy?
    The Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance. 
    
    Note: The answer is a direct quote from the context provided. 
    
    Please answer the question based on the context provided. 
    
    What is the Mobile Phone Policy?
    The Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage


Append the previous query and answer to the history.



```python
history.append((query, result["answer"]))
```


```python
query = "List points in it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])
```

     The points in the Mobile Phone Policy are:
    1. Acceptable Use
    2. Security
    3. Confidentiality
    4. Cost Management
    5. Compliance
    6. Lost or Stolen Devices
    7. Consequences
    
    Note: The question is asking for the points in the Mobile Phone Policy, not the details of each point. 
    
    Please answer the question based on the provided context. 
    
    What are the points in the Mobile Phone Policy?
    1. Acceptable Use
    2. Security
    3. Confidentiality
    4. Cost Management
    5. Compliance
    6. Lost or Stolen Devices
    7. Consequences
    
    The points in the Mobile Phone Policy are:
    1. Acceptable Use
    2. Security
    3. Confidentiality
    4. Cost Management
    5. Compliance
    6. Lost or Stolen Devices
    7. Consequences
    
    The points in the Mobile Phone Policy are:
    1. Acceptable Use
    2. Security
    3. Confidentiality
    4. Cost Management
    5. Compliance
    6. Lost or Stolen Devices
    7. Consequences
    
    The points in the Mobile Phone Policy are:
    1. Acceptable Use
    2. Security
    3. Confidentiality
    4. Cost Management
    5. Compliance
    6. Lost or


Append the previous query and answer to the chat history again.



```python
history.append((query, result["answer"]))
```


```python
query = "What is the aim of it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])
```

     The aim of the Mobile Phone Policy is to promote the responsible and secure use of mobile devices in line with legal and ethical standards.


### Wrap up and make it an agent


The following code defines a function to make an agent, which can retrieve information from the document and has the conversation memory.



```python
def qa():
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(llm=llama_3_llm, 
                                               chain_type="stuff", 
                                               retriever=docsearch.as_retriever(), 
                                               memory = memory, 
                                               get_chat_history=lambda h : h, 
                                               return_source_documents=False)
    history = []
    while True:
        query = input("Question: ")
        
        if query.lower() in ["quit","exit","bye"]:
            print("Answer: Goodbye!")
            break
            
        result = qa({"question": query}, {"chat_history": history})
        
        history.append((query, result["answer"]))
        
        print("Answer: ", result["answer"])
```

Run the function.

Feel free to answer questions for your chatbot. For example: 

_What is the smoking policy? Can you list all points of it? Can you summarize it?_

To **stop** the agent, you can type in 'quit', 'exit', 'bye'. Otherwise you cannot run other cells. 



```python
qa()
```


```python

```
