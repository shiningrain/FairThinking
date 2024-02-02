import time
import os
import json
import random
# random.seed(0)
import sys
sys.path.append('./utils')
from agent import *
from role_generator import generate_roles
from inferenceLLM import load_llama_model
from debate_utils import *
import pickle
import argparse
import ast
import shutil
import torch
import csv

def read_topics(topic_path):
    topic_list=[]
    with open(topic_path, mode ='r')as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            topic_list.append(line[0])

    return topic_list

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ./demo/Comparative_Questions/0/question.pkl
    parser.add_argument("-tp", "--topic_path", default='./demo/Targeted_Open-Ended_Questions/topics.csv', type=str, help="Input file, can be 'pkl' or 'csv'")
    parser.add_argument("-od", "--output_dir", default='./demo_result/Type2/0', type=str, help="Output file dir")
    parser.add_argument("-cp", "--config_path", default='./config/debate_config.json', type=str, help="config")
    parser.add_argument("-at", "--attribute", default='generator', type=str, help="the role of each member:generator/debater(ablation, no role)/identity(ablation, only identity)")#-Asian woman-Latin American man
    parser.add_argument("-md", "--model", default='mistral', type=str, help="gpt(GPT-3.5-Turbo)/llama(llama2)/mistral(mistral)/gpt4(GPT-4-1106-preview)")
    parser.add_argument("-gl", "--gpu_list", default='1-1-1-1-1-1', type=str, help="gpu list, whose length should not less than member number+1")
    parser.add_argument("-rp", "--repeat", default='1', type=str, help="repeat experiment")
    parser.add_argument("-jr", "--jury", default='6', type=str, help="max number of jury members in debate")
    return parser.parse_args()


if __name__ == "__main__":

    current_script_path = os.path.abspath(__file__)
    args = parse_args()
    topic_path = args.topic_path
    topic_dir=os.path.dirname(args.topic_path)
    if args.jury!='None' and args.jury!=None:
        jury=int(args.jury)
    else:
        jury=None
        jury_list=[]
    onlyjury=False

    if args.model=='llama' or args.model=='vicuna' or args.model=='mistral':
        gpu_list=[int(i) for i in args.gpu_list.split('-')]
    
    config_path = os.path.join(os.path.dirname(topic_path),'config.pkl')
    save_dir=args.output_dir
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if '.pkl' in args.topic_path:
        with open(args.topic_path, 'rb') as f:
            topic_result = pickle.load(f)
        total_topic=[]
        for tmp_topic in topic_result:
            total_topic+=tmp_topic
    elif '.csv' in args.topic_path:
        total_topic=read_topics(args.topic_path)
    else:
        print('not support input!!')
        os._exit(0)

    model_misc=None
    if args.model=='llama':
        model,tokenizer=load_llama_model('7b',device=gpu_list[0])
        model_misc=(model,tokenizer)
    elif args.model=='mistral':
        model,tokenizer=load_llama_model('mistral',device=gpu_list[0])
        model_misc=(model,tokenizer)
    for i in range(len(total_topic)):
        debate_topic=total_topic[i]
        print(debate_topic)
        for j in range(int(args.repeat)):
            tmp_save_dir=os.path.join(topic_dir,f'{i}-multi_role-result')
            tmp_result_dir=os.path.join(save_dir,f'{i}-multi_role-result')
            print(tmp_result_dir)
            # os._exit(0)
            if not os.path.exists(tmp_save_dir):
                os.makedirs(tmp_save_dir)
            if not os.path.exists(tmp_result_dir):
                os.makedirs(tmp_result_dir)
            # # If have already generated results: skip
            # if len(os.listdir(tmp_result_dir))>2:
            #     continue
            if jury!=None:# use jury to evaluate
                if os.path.exists(os.path.join(tmp_result_dir,'result-0-jury.pkl')):
                    continue
                elif os.path.exists(os.path.join(tmp_result_dir,'result-0.pkl')): # if has debate results ,then only use jury
                    onlyjury=True
                else:
                    onlyjury=False
            # not use jury to evaluate
            if jury==None and os.path.exists(os.path.join(tmp_result_dir,'result-0.pkl')):
                continue

            if args.attribute=='generator' or args.attribute=='identity':
                generated_role_path=os.path.join(tmp_save_dir,'attribute_list.pkl')#tmp_role_dir
                if os.path.exists(generated_role_path):
                    with open(generated_role_path, 'rb') as f:#input,bug type,params
                        attribute_list = pickle.load(f)
                else:
                    attribute_list=generate_roles(debate_topic)
                    with open(generated_role_path, 'wb') as f:
                        pickle.dump(attribute_list, f)
                if args.attribute=='identity':
                    attribute_list=[attri.split(' Your personality')[0] for attri in attribute_list]
                    print(attribute_list)
            elif args.attribute=='debater':
                attribute_list=["You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer.\n" for k in range(4)]
            else:
                print('No Attribute!')
                os._exit(0)

            if jury!=None:
                generated_jury_path=os.path.join(tmp_save_dir,'jury_list.pkl')#tmp_role_dir
                if os.path.exists(generated_jury_path):
                    with open(generated_jury_path, 'rb') as f:#input,bug type,params
                        jury_list = pickle.load(f)
                    if len(jury_list)!=int(args.jury):
                        jury_list=generate_roles(debate_topic,template_path='./utils/prompt_template_automatic_jury',max_role=6,role='jury')
                        with open(generated_jury_path, 'wb') as f:
                            pickle.dump(jury_list, f)
                else:
                    jury_list=generate_roles(debate_topic,template_path='./utils/prompt_template_automatic_jury',max_role=6,role='jury')
                    with open(generated_jury_path, 'wb') as f:
                        pickle.dump(jury_list, f)
            member=int(len(attribute_list))

            config = json.load(open(args.config_path, "r"))
            config['debate_topic'] = debate_topic
            
            print(f'start {tmp_result_dir}')
            print(attribute_list)
            if args.model=='gpt':
                debate = Debate(model_name='gpt-3.5-turbo',num_players=member+2, max_round=3,openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=5,attribute_list=attribute_list,gpu_list=None,jury=jury,jury_list=jury_list,repeat=j,only_jury=onlyjury,model_misc=model_misc)
            elif args.model=='gpt4':
                debate = Debate(model_name='gpt-4-1106-preview',num_players=member+2, max_round=3,openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=5,attribute_list=attribute_list,gpu_list=None,jury=jury,jury_list=jury_list,repeat=j,only_jury=onlyjury,model_misc=model_misc)
            elif args.model=='llama':
                debate = Debate(model_name='llama',num_players=member+2, max_round=3,openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=5,attribute_list=attribute_list,gpu_list=gpu_list,jury=jury,jury_list=jury_list,repeat=j,only_jury=onlyjury,model_misc=model_misc)
            elif args.model=='mistral':
                debate = Debate(model_name='mistral',num_players=member+2, max_round=3,openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=5,attribute_list=attribute_list,gpu_list=gpu_list,jury=jury,jury_list=jury_list,repeat=j,only_jury=onlyjury,model_misc=model_misc)
            debate.run(tmp_result_dir)
            time.sleep(3)
            if jury!=None:
                debate.init_jury()
                debate.jury_evaluate(tmp_result_dir)
            if args.model=='llama' or args.model=='mistral':
                del debate
                torch.cuda.empty_cache()
    print(f'Finish-{args.topic_path}')
    # time.sleep(60)