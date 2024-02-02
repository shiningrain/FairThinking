import openai
import configparser
import os
import numpy as np
import time
import pickle
import sys




def read_config(name='OPENAI',path='./utils/config.cfg'):
    config = configparser.RawConfigParser()
    config.read(path)
    details_dict = dict(config.items(name))
    return details_dict
 
def get_reason_list(prompt_sentence):
    prompt_sentence='The following statement is an answer and explanation to a certain question. Please extract the reasons mentioned in it and organize them into a list, for example [xxx,xxx,...]. Please strictly output in Python List format, do not output irrelevant content. \n\nThe statement is: \n'+prompt_sentence
    answer=query_gpt(version='gpt-3.5-turbo',question=prompt_sentence,sleep=3,system_prompt="You are a professional editor.")# ['message'].content
    return answer

def query_gpt(version='gpt-3.5-turbo',question=None,sleep=3,system_prompt="You are a professional debate organizer"):
    config=read_config(name='OPENAI')
    # openai.organization = config['orgid']
    api_key = config['key']
    
    messages=[
    {"role": "system", "content": system_prompt},#"You are a helpful assistant."},
    {"role": "user", "content": question}
    ]
    # client = OpenAI(api_key=config['key'])# for 1.x version
    # resp = client.chat.completions.create( 
    #     model=version,
    #     messages=messages)
    response = openai.ChatCompletion.create(# for 0.28 version
                    model=version,
                    messages=messages,
                    max_tokens=500,
                    api_key=api_key,
                )
    resp = response['choices'][0]['message']['content']
    time.sleep(sleep)
    return resp

def extract_response_list(answer,max_num=4):
    try:
        response_list=eval(answer)
    except:
        if not ('[' in answer and ']' in answer):
            print(answer)
            print('Unknown format!')
            return []
        else:
            if answer[0]=='[' and answer[-1]==']':
                print('Failed to convert result!')
                return []

            # Find the first '['
            try:
                response_list=eval(answer[answer.find('['):answer.rfind(']')+1])
            except:
                return []
    if len(response_list)>max_num:
        print('Too many roles!')
        return response_list[:max_num]
    return response_list

def extract_response_dict(answer,key_list):
    try:
        response_dict=eval(answer)
    except:
        if not ('{' in answer and '}' in answer):
            print(answer)
            print('Unknown format!')
            return {}
        else:
            if answer[0]=='{' and answer[-1]=='}':
                print('Failed to convert result!')
                return {}

            # Find the first '['
            response_dict=eval(answer[answer.find('{'):answer.rfind('}')+1])
    for key in key_list:
        if key not in response_dict:
            print(f'Miss a requried key: {key}')
            return {}
    return response_dict

def get_llm_answer(tmp_prompt,model,model_misc,sleep=3,system_prompt="You are a professional debate organizer"):
    if model=='gpt3.5':
        answer=query_gpt(question=tmp_prompt,sleep=sleep,system_prompt=system_prompt)
    elif model=='gpt4':
        answer=query_gpt(version='gpt-4-1106-preview',question=tmp_prompt,sleep=sleep,system_prompt=system_prompt)
    elif model=='vicuna':
        input_text='[system]:{}\n[user]:{}\n[assistant]:'.format(system_prompt,tmp_prompt)
        answer=inference_vicuna_model(model_misc[0],model_misc[1],input_text,max_new_tokens=1000)
    elif model=='llama':
        input_text='[system]:{}\n[user]:{}\n[assistant]:'.format(system_prompt,tmp_prompt)
        answer=inference_llama_model(model_misc[0],model_misc[1],[input_text],max_new_tokens=1000)
    return answer

def get_identity(topic,
                 role_text_list,
                 max_role=4,
                 count=5,
                 string1='You are acting as a debater from the group {###identity}. Your personality is {###MBTI}.',
                 role='debater',
                 model=None,
                 model_misc=None):
    # role is debater or jury 
    if role =='debater':
        tmp_prompt='The topic of the debate is `{}`. You need to choose the most suitable stakeholders from different groups with different identities to participate in the debate. Make sure that the diversity of views will be considerred in debate. The total number of characters SHOULD BE {}. Please give the group identity and MBTI personality participating in the debate, use the list [[\"identity1\",\"MBTI1\"],...] to answer directly, do not answer other content.'
    elif role=='jury':
        tmp_prompt='The topic of the debate is `{}`. You need to choose the most suitable stakeholders from different groups with different identities as jurors to judge the final answer of the debate. Make sure that the diversity of views will be considerred in debate. The total number of characters SHOULD BE {}. Please give the group identity and MBTI personality participating in the debate, use the list [[\"identity1\",\"MBTI1\"],...] to answer directly, do not answer other content.'
    tmp_prompt=tmp_prompt.format(topic,max_role)
    answer=get_llm_answer(tmp_prompt,model,model_misc)
    # answer=query_gpt(question=tmp_prompt)# ['message'].content
    answer_list=extract_response_list(answer,max_role)
    while answer_list==[] and count>0: # if no valid results
        count-=1
        answer=get_llm_answer(tmp_prompt,model,model_misc)
        # answer=query_gpt(question=tmp_prompt)# ['message'].content
        answer_list=extract_response_list(answer,max_role)
    for tmp_anwser in answer_list:
        tmp_string=string1.replace('{###identity}',tmp_anwser[0])
        tmp_string=tmp_string.replace('{###MBTI}',tmp_anwser[1])
        role_text_list.append([tmp_string,tmp_anwser])
    return role_text_list

def get_concept(topic,
                role_text_list,
                count=5,
                string1='The person you admire most is {##celebrity}({##celebrity_description}). The concept you believe in is `{##concept}`, and `{##slogan}` is your slogan and belief.',
                aligned=True,
                role='debater',
                model=None,
                model_misc=None):
    '''if aligned=True, the concept and slogan should be aligned corresponding identities, else random'''
    max_role=len(role_text_list)
    if role=='debater':
        requirement='to participate in the debate. The identities of debaters'
    elif role=='jury':
        requirement='as jurors to judge the debate answer. The identities of debaters of jurors'
    if aligned:
        identity_string=','.join([i[1][0]+'-'+i[1][1] for i in role_text_list])
        tmp_prompt=f'The topic of the debate is `{topic}`. You need to choose the most suitable people with different concepts {requirement} are {identity_string}. Please tell me 1 celebrity who might be icons for each identity and his/her one-sentence description, use the Python List format to answer, like  `[[\"identity1\",\"celebrity1\",\"description1\"],...]` Do not answer other content.'
        answer1=get_llm_answer(tmp_prompt,model,model_misc)
        # answer1=query_gpt(question=tmp_prompt)# ['message'].content
        answer_list1=extract_response_list(answer1,max_role)
        while answer_list1==[] and count>0: # if no valid results
            count-=1
            answer1=get_llm_answer(tmp_prompt,model,model_misc)
            # answer1=query_gpt(question=tmp_prompt)# ['message'].content
            answer_list1=extract_response_list(answer1,max_role)
        tmp_prompt=f'The topic of the debate is `{topic}`. You need to choose the most suitable people with different concepts {requirement} are {identity_string}. For each identity, please choose an ideological or political concept that they may believe in and a corresponding slogan. Use the List format like [[\"identity1\",\"concept1\",\"slogan1\"],...] to show the answer directly, each element in the list only consists of a concept and a slogan, and do not answer other content.'
        answer2=get_llm_answer(tmp_prompt,model,model_misc)
        # answer2=query_gpt(question=tmp_prompt)# ['message'].content
        answer_list2=extract_response_list(answer2,max_role)
        while answer_list2==[] and count>0: # if no valid results
            count-=1
            answer2=get_llm_answer(tmp_prompt,model,model_misc)
            # answer2=query_gpt(question=tmp_prompt)# ['message'].content
            answer_list2=extract_response_list(answer2,max_role)
        answer_list=[answer_list1[i][1:]+answer_list2[i][1:] for i in range(len(answer_list1))]
    else:
        role_num=len(role_text_list)
        tmp_prompt=f'Please tell me {role_num} celebrities who might be icons for one people and his/her one-sentence description. Also, please give me {role_num} ideological or political concepts that some people may believe in and the corresponding slogans. Use the List format like [[\"celebrity\",\"description\",\"concept\",\"slogan\"],...] to answer directly, do not answer other content.'
        answer=get_llm_answer(tmp_prompt,model,model_misc)
        # answer=query_gpt(question=tmp_prompt)# ['message'].content
        answer_list=extract_response_list(answer,max_role)
        while answer_list==[] and count>0: # if no valid results
            count-=1
            answer=get_llm_answer(tmp_prompt,model,model_misc)
            # answer=query_gpt(question=tmp_prompt)# ['message'].content
            answer_list=extract_response_list(answer,max_role)
    for i in range(len(answer_list)):
        tmp_anwser=answer_list[i]
        tmp_string=string1.replace('{##celebrity}',tmp_anwser[0])
        tmp_string=tmp_string.replace('{##celebrity_description}',tmp_anwser[1])
        tmp_string=tmp_string.replace('{##concept}',tmp_anwser[2])
        tmp_string=tmp_string.replace('{##slogan}',tmp_anwser[3])
        role_text_list[i][0]+=tmp_string
        role_text_list[i].append(tmp_anwser)
    return role_text_list

def generate_experience(topic,
                        role_text_list,
                        count=5,
                        string1='{#growth_experience} {#group_relation}',
                        role='debater',
                        model=None,
                    model_misc=None):
    if role=='debater':
        requirement='debater'
    elif role=='jury':
        requirement='juror'
    for r in range(len(role_text_list)):
        role_text=role_text_list[r]
        role=role_text[0]
        tmp_role=role.replace(f'You are acting as a {requirement}','A person is').lower().replace('your','his/her').replace('you','he/she')
        tmp_prompt=f'For the given character: {tmp_role}\n. For the above character, please design the growth experience (two sentences) based on the real world, and write about the current status  (two sentences) in his/her group based on their identity and personality.'+'Now please output your answer in JSON format, with the format as follows: {\"Growth Experience\": \"\", \"Current Status\": \"\"}\n Do not answer other content.'
        answer=get_llm_answer(tmp_prompt,model,model_misc,sleep=3,system_prompt="You are a realistic writer")
        # answer=query_gpt(version='gpt-3.5-turbo',question=tmp_prompt,sleep=3,system_prompt="You are a realistic writer")# ['message'].content
        answer_dict=extract_response_dict(answer,key_list=['Growth Experience','Current Status'])
        while answer_dict=={} and count>0: # if no valid results
            count-=1
            answer=get_llm_answer(tmp_prompt,model,model_misc,sleep=3,system_prompt="You are a realistic writer")
            # answer=query_gpt(version='gpt-3.5-turbo',question=tmp_prompt,sleep=3,system_prompt="You are a realistic writer")# ['message'].content
            answer_dict=extract_response_dict(answer,key_list=['Growth Experience','Current Status'])
        tmp_string=string1.replace('{#growth_experience}',answer_dict['Growth Experience'])
        tmp_string=tmp_string.replace('{#group_relation}',answer_dict['Current Status'])
        role_text_list[r][0]+=tmp_string
        role_text_list[r].append(answer_dict)
    return role_text_list

def polish_prompt(prompt_sentence):
    prompt_sentence='For a given text, you need to modify its punctuation and the connection between sentences to make the text smoother without any change in semantics. \n\nThe text is: \n'+prompt_sentence
    answer=query_gpt(version='gpt-3.5-turbo',question=prompt_sentence,sleep=3,system_prompt="You are a professional magazine editor.")# ['message'].content
    return answer


def generate_roles(topic,
                   max_role=4,
                   template_path='./utils/prompt_template_automatic',
                   polish=False, # useless, inefficient
                   aligned=True,
                   role='debater',
                   model='gpt3.5',
                   device_num=None):
    role_text_list=[]
    # step 1: load template
    f=open(template_path,'r')
    template_list=f.readlines()
    f.close()
    if model=='gpt3.5' or model=='gpt4':
        model_misc=None
    # step 2: get diverse identities/race/gender
    role_text_list=get_identity(topic,role_text_list,string1=template_list[0].strip(),max_role=max_role,role=role,model=model,model_misc=model_misc)
    # step 3: add concepts/view/slogan
    role_text_list=get_concept(topic,role_text_list,string1=template_list[1].strip(),aligned=aligned,role=role,model=model,model_misc=model_misc)
    # step 3: add grow experience, polish the role.
    role_text_list=generate_experience(topic,role_text_list,string1=template_list[2].strip(),role=role,model=model,model_misc=model_misc)

    pure_role_list=[]
    for r in range(len(role_text_list)):
        if polish:
            role_text_list[r][0]=polish_prompt(role_text_list[r][0])
        role_text_list[r][0]+=template_list[3].strip()
        pure_role_list.append(role_text_list[r][0])

    return  pure_role_list
