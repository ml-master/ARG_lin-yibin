# chat with ollama
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
import pandas as pd
import json
import random


TEMPLATE = '''

Question: {question}

Answer: {sepecific_prompt}
'''

PROMPT_TEMPLATE = PromptTemplate(
    template=TEMPLATE,
    input_variables=["question","sepecific_prompt"],
)

def create_ollama_connection(model: str, base_url="http://lake-gpu-0:11434") -> Ollama:
    return Ollama(model=model, base_url=base_url)

def format_prompt(data: pd.DataFrame, question: str) -> str:
    return TEMPLATE.format(data=data, question=question)

def ask_ollama(model: str, prompt: str) -> str:
    ollama = create_ollama_connection(model)
    result = ''
    for chunks in ollama.stream(prompt):
        result += chunks
    return result

# def build_dataframe_prompt(data: pd.DataFrame, question: str):
#     data_buffer = ''
#     for index, row in data.iterrows():
#         data_buffer += str(row.to_dict()) + '\n'
#     return PROMPT_TEMPLATE.format(data=data_buffer, question=question)

# def read_json_prompt(data:dict, question:str):
    

def main():
    with open("gossipcop_v3-4_story_based_fake.json",'r',encoding='utf-8')as f:
        o_data = json.load(f)
    
    question = "Given the following message, predict its veracity. If it is more likely to be a real message, return 1; otherwise, return 0. give the 1 or 0 at the end of the answer. Please refrain from providing ambiguous assessments such as undetermined: "
    textual_description = "Let’s think from the perspective of textual description."
    commonsense = "Let’s think from the perspective of commonsense."
    i:int = 0
    
    new_dataset = []
    for i in range(0,5000):
        random_item = random.choice(list(o_data.items()))
        item = random_item[1]
        print(random_item[0])
        news_content = item["origin_text"]
        c_q = PROMPT_TEMPLATE.format(question = question+news_content ,sepecific_prompt = commonsense)
        t_q = PROMPT_TEMPLATE.format(question = question+news_content ,sepecific_prompt = textual_description)
        c_result = ask_ollama('llama3',c_q)
        t_result = ask_ollama('llama3',t_q)
        
        
        if item["origin_label"] == "fake":
            label = 0
        else:
            label = 1
        
        
        if t_result[-1] == "1":
            td_pred = 1
        elif t_result[-1] == "0":
            td_pred = 0
        else:
            continue
        
        if td_pred == label:
            td_acc = 1
        else:
            td_acc = 0
            
        if c_result[-1] == "1":
            cs_pred = 1
        elif c_result[-1] == "0":
            cs_pred = 0
        else:
            continue
        
        if cs_pred == label:
            cs_acc = 1
        else:
            cs_acc = 0
            
        data = {"content":news_content,
                "label":label,
                "time":None,
                "source_id":i,
                "td_rationale":t_result,
                "td_pred":td_pred,
                "td_acc":td_acc,
                "cs_rationale":c_result,
                "cs_pred":cs_pred,
                "cs_acc":cs_acc,
                "split":"train"
                }
        new_dataset.append(data)
        if i%100:
            with open("new_dataset.json",'w',encoding='utf-8')as f:
                f.write(json.dumps(new_dataset))
                    
    with open("new_dataset.json",'w',encoding='utf-8')as f:
        f.write(json.dumps(new_dataset))
    

    
    # df = read_git_table('resources/git-tables/abstraction_csv_licensed/00-01_56.csv')
    # langchain_prompt = build_dataframe_prompt(df, 'select all the data rows whose Type is story and delete the ID column and delete the Created column.')
    # print(len(langchain_prompt))
    # result = ask_ollama('llama3', prompt)
    # print(result)

if __name__ == "__main__":
    main()
    