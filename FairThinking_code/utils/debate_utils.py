import time
import os
import sys
from agent import Agent
import pickle
import sys
from inferenceLLM import load_llama_model,inference_llama_model
import torch
import configparser



def read_config(name='OPENAI',path='./utils/config.cfg'):
    config = configparser.RawConfigParser()
    config.read(path)
    details_dict = dict(config.items(name))
    return details_dict
openai_api_key = read_config()['key']

# NAME_LIST=[
#     "Affirmative side",
#     "Negative side",
#     "Moderator",
# ]

def get_total_reason_dict(reason_dict_string):
    try:
        reason_dict=eval(reason_dict_string)
    except:
        if reason_dict_string[-1]=='.':
            reason_dict_string=reason_dict_string[:-1]
        if '{' in reason_dict_string and '}' in reason_dict_string:
            reason_dict_string=reason_dict_string[:reason_dict_string.index('}')+1]

        if reason_dict_string[-1]!='}':
            if reason_dict_string[-1]==']':
                reason_dict_string+='}'
            elif reason_dict_string[-1]=='\"':
                reason_dict_string+=']}'
            else:
                reason_dict_string+='\"]}'
        try:
            reason_dict=eval(reason_dict_string)
        except:
            reason_dict={}
    return reason_dict

def load_attribute(attribute,config_path):
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:#input,bug type,params
            config = pickle.load(f)
        # print(config)
        return config
    else:
        if attribute=='sex':
            return ['male','female']
        elif attribute=='age':
            return ['young','elder']
        else:
            return attribute.split('-')

def get_role_prompt(prompt_string,role_path):
    if os.path.exists(role_path):
        f=open(role_path,'r')
        role_string=''.join(f.readlines())
        f.close()
    else:
        role_string=role_path
    prompt_string=prompt_string.replace('##insert_role##',role_string)
    return prompt_string


class DebatePlayer(Agent):
    def __init__(self,
                 model_name: str,
                 name: str,
                 temperature:float,
                 openai_api_key: str,
                 sleep_time: float,
                 gpu=None,
                 model_misc=None
                 ) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """

        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key
        if model_name=='llama' or model_name=='vicuna' or model_name=='mistral':
            # self.model,self.tokenizer=load_llama_model(model_name='7b',device=gpu)
            (self.model,self.tokenizer)=model_misc


class Debate:
    def __init__(self,
            model_name: str='gpt-3.5-turbo', 
            temperature: float=0, 
            num_players: int=3, 
            openai_api_key: str=None,
            config: dict=None,
            max_round: int=3,
            sleep_time: float=0,
            repeat: int=1,
            attribute_list=None,
            gpu_list=None,
            jury=None,
            jury_list=[],
            only_jury=False,
            model_misc=None
        ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            openai_api_key (str): As the parameter name suggests
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.num_member = num_players-2
        self.num_jury=jury
        self.jury_list=jury_list
        self.max_round = max_round
        self.reason_list=[]
        self.clerk_results=''
        self.repeat=repeat
        self.model_misc=model_misc
        self.gpu_list=gpu_list
        if self.gpu_list==None:
            self.gpu_list=[0 for i in range(self.num_players)]
        self.answer_list = {}
        self.openai_api_key = openai_api_key
        # tmp='The results of each member is summarized in JSON format.'
        # for i in range(num_players-1):
        #     tmp+=f'Member {i+1} side provides the answer and reason: ##ans_{i+1}##\n\n'
        tmp=''
        for i in range(self.num_member):
            tmp+=f'Member {i+1} side provides the answer: ##ans_{i+1}##\n\n'
        config['moderator_prompt']=config['moderator_prompt'].replace('##iterate_add##',tmp)
        config['clerk_prompt']=config['clerk_prompt'].replace('##iterate_add##',tmp)
        # config['judge_prompt_last1']=config['judge_prompt_last1'].replace('##iterate_add1##',tmp)
        # tmp=''
        # for i in range(self.max_round):
        #     tmp+=f'In Round {i+1}, the reasons provided by debaters are `##ans0_{i+1}##`\n\n'
        # config['judge_prompt_last2']=config['judge_prompt_last2'].replace('##iterate_add##',tmp)
        self.config = config
        self.only_jury = only_jury
        
        self.sleep_time = sleep_time
        self.attri=attribute_list
        # assert(len(self.attri)==self.num_member)
        self.init_prompt() # replace topic in the prompts

        # creat&init agents
        if not self.only_jury:
            self.creat_agents() # generate the agensts with different roles

            self.init_agents() # initiate agent response


    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("clerk_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")
        prompt_replace('jury_meta_prompt')

    def creat_agents(self):
        NAME_LIST=[]
        for i in range(self.num_member):
            NAME_LIST.append(f'Member_{i+1}')
        NAME_LIST.append('Moderator')
        NAME_LIST.append('Clerk')
        # creates players
        if self.model_misc==None:
            if self.model_name=='llama':
                torch.cuda.empty_cache()
                model,tokenizer=load_llama_model(model_name='7b',device=self.gpu_list[0])
                model_misc=(model,tokenizer)
            elif self.model_name=='vicuna':
                torch.cuda.empty_cache()
                model,tokenizer=load_llama_model(model_name='vicuna',device=self.gpu_list[0])
                model_misc=(model,tokenizer)
            elif self.model_name=='mistral':
                torch.cuda.empty_cache()
                model,tokenizer=load_llama_model(model_name='mistral',device=self.gpu_list[0])
                model_misc=(model,tokenizer)
            else:
                model_misc=self.model_misc
        else:
            model_misc=self.model_misc
        self.players = [
            DebatePlayer(model_name=self.model_name,
            name=NAME_LIST[n],
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time,
            gpu=self.gpu_list[0],model_misc=model_misc) for n in range(len(NAME_LIST))
        ]
        self.judge_player=DebatePlayer(model_name=self.model_name,
                                    name='Judge',
                                    temperature=self.temperature,
                                    openai_api_key=self.openai_api_key,
                                    sleep_time=self.sleep_time,
                                    gpu=self.gpu_list[0],model_misc=model_misc)
        self.member=[]
        for i in range(self.num_member):
            self.member.append(self.players[i])
        self.moderator=self.players[-2]
        self.clerk=self.players[-1]


    def init_agents(self):

        for i in range(self.num_member):
            new_prompt=get_role_prompt(self.config['player_meta_prompt'],self.attri[i])
            self.member[i].set_meta_prompt(new_prompt)
            # self.member[i].set_meta_prompt(self.config['player_meta_prompt'].replace("##attribute##", self.attri[i]))
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])
        self.clerk.set_meta_prompt(self.config['clerk_meta_prompt'])  
        
        if not self.only_jury:
            # start: first round debate, state opinions
            print(f"===== Debate Round-1 =====\n")
            self.answer_list[1]=[]
            tmp_answer_list=self.answer_list[1]
            for i in range(self.num_member):
                if i==0:
                    self.member[i].add_event(self.config['affirmative_prompt'])
                else:
                    self.member[i].add_event(self.config['negative_prompt'].replace('##aff_ans##', tmp_answer_list[i-1]))# argue the last one's point
            # for i in range(self.num_member):
                tmp_answer_list.append(self.member[i].ask(300))
            # for i in range(self.num_member):
                self.member[i].add_memory(tmp_answer_list[i])
            self.config['base_answer'] = tmp_answer_list[0]

            tmp_event=self.config['clerk_prompt'].replace('##round##', 'first')
            # tmp_event=tmp_event.replace('##format##', 'Now please output your answer in json format whose keys are different \"Answer\" in the dabate and values are corresponding \"Reason\". The format is like follows: {\"Answer1: xxx\": \"[xxx,xxx,..]\", \"Answer2: xxx\": \"[xxx,xxx,..]\"}')
            if self.model_name=='vicuna':
                tmp_event=tmp_event.replace('##format##', 'Now please output your answer in JSON format and each key is one answer in the dabate and each value is a list of corresponding reason. The format is like follows: {\"Pros: xxx\": \"[xxx,xxx,..]\", \"Cons: xxx\": \"[xxx,xxx,..]\", \"Neutral: xxx\": \"[xxx,xxx,..]\"}')
            tmp_event=tmp_event.replace('##format##', 'Now please output your answer in JSON format whose keys are different \"Answer\" in the dabate and values MUST be list of corresponding \"Reason\". The format is like follows: {\"Pros: xxx\": \"[xxx,xxx,..]\", \"Cons: xxx\": \"[xxx,xxx,..]\", \"Neutral: xxx\": \"[xxx,xxx,..]\"}')
            for i in range(self.num_member):
                tmp_event=tmp_event.replace(f'##ans_{i+1}##',tmp_answer_list[i])
            self.clerk.add_event(tmp_event)
            self.mod_ans = self.clerk.ask(700)
            self.clerk_results=self.mod_ans
            self.clerk.add_memory(self.mod_ans)

            tmp_event=self.config['moderator_prompt'].replace('##round##', 'first')
            # tmp_event=tmp_event.replace('##format##', 'Now please output your answer in json format whose keys are different \"Answer\" in the dabate and values are corresponding \"Reason\". The format is like follows: {\"Answer1: xxx\": \"[xxx,xxx,..]\", \"Answer2: xxx\": \"[xxx,xxx,..]\"}')
            for i in range(self.num_member):
                tmp_event=tmp_event.replace(f'##ans_{i+1}##',tmp_answer_list[i])
            self.moderator.add_event(tmp_event)
            self.mod_ans = self.moderator.ask(500)
            self.moderator.add_memory(self.mod_ans)
            try:
                self.mod_ans = eval(self.mod_ans)   
            except:
                print('Format error!')
                # self.mod_ans={'Reason':self.mod_ans}
            self.reason_list.append(self.mod_ans)
        

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def save_answser(self,save_dir,misc=''):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file=os.path.join(save_dir,f'result_log-{self.repeat}{misc}')
        f=open(save_file,'w')
        try:
            f.writelines("\n\n===== Debate Done! =====")
            f.writelines("\n----- Debate Topic -----")
            f.writelines(str(self.config["debate_topic"]))
            f.writelines("\n----- Base Answer -----")
            f.writelines(str(self.config["base_answer"]))
            f.writelines("\n----- Debate Answer -----")
            f.writelines(str(self.config["Answer"]))
            f.writelines("\n----- Debate Reason -----")
            f.writelines(str(self.config["Reason"]))
        except:
            pass
        f.close()
        save_path=os.path.join(save_dir,f'result-{self.repeat}{misc}.pkl')

        result_dict={}
        result_dict["debate_topic"]=self.config["debate_topic"]
        result_dict["base_answer"]=self.config["base_answer"]
        result_dict["debate_answer"]=self.config["Answer"]

        result_dict["Reason"]=self.config["Reason"]
        result_dict["All_Reasons"]=self.clerk_results
        # result_dict["Reason_list"]=get_reason_list(self.config["Reason"])#TODO: tomorrow process together
        
        # result_dict["memory_list"]=[mem.memory_lst for mem in self.member]
        try:
            result_dict["player_memory_list"]=[[mem.name,mem.memory_lst] for mem in self.players]
        except:
            pass
        try:
            result_dict["jury_memory_list"]=[[mem.name,mem.memory_lst] for mem in self.jury_members]
            result_dict["jury_answer"]=self.config['JuryAnswer']
            result_dict["jury_count"]=self.config['JuryNumber']
        except:
            pass
        # result_dict["moderator_memory"]=self.moderator.memory_lst
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)


    def run(self,save_dir):
        '''
        moderator_result = ast.literal_eval(self.moderator.memory_lst[-1]['content'])
        if moderator_result['Reason']!='':
            argu_content=moderator_result['Reason']
        else:
            xxx
        '''
        if self.only_jury:
            print('Only jury, Skipping debate!')
            return 0
        
        for round in range(self.max_round - 1):
            time.sleep(3)
            # if self.mod_ans["Answer"] == '???':# not early stop now
            # if self.mod_ans["Answer"] != '':# or ('continue' not in self.mod_ans["Answer"] or 'further' not in self.mod_ans["Answer"])
            #     break
            # else:
            print(f"===== Debate Round-{round+2} =====\n")
            self.answer_list[round+2]=[self.answer_list[round+2-1][-1]]
            tmp_answer_list=self.answer_list[round+2]
            for i in range(self.num_member):
                self.member[i].add_event(self.config['debate_prompt'].replace('##oppo_ans##', tmp_answer_list[i-1])) # argue with the last one
                if i==0:
                    tmp_answer_list[0]=self.member[i].ask(300)
                else:
                    tmp_answer_list.append(self.member[i].ask(300))
                self.member[i].add_memory(tmp_answer_list[i])

            tmp_event=self.config['clerk_prompt'].replace('##round##', self.round_dct(round+2))
            # tmp_event=tmp_event.replace('##format##', 'Now please output your answer in json format whose keys are different \"Answer\" in the dabate and values are corresponding \"Reason\" list. The format is like follows: {\"Answer1: xxx\": \"[xxx,xxx,..]\", \"Answer2: xxx\": \"[xxx,xxx,..]\"}\nEach element in the \"Reason\" list is a complete sentence.')
            if self.model_name=='vicuna':
                tmp_event=tmp_event.replace('##format##', 'Now please output your answer in JSON format and each key is one answer in the dabate and each value is a list of corresponding reason`. The format is like follows: {\"Pros: xxx\": \"[xxx,xxx,..]\", \"Cons: xxx\": \"[xxx,xxx,..]\", \"Neutral: xxx\": \"[xxx,xxx,..]\"}')
            tmp_event=tmp_event.replace('##format##', 'Now please output your answer in json format whose keys are different \"Answer\" in the dabate and values MUST be corresponding \"Reason\". The format is like follows: {\"Pros: xxx\": \"[xxx,xxx,..]\", \"Cons: xxx\": \"[xxx,xxx,..]\", \"Neutral: xxx\": \"[xxx,xxx,..]\"}')
            for i in range(self.num_member):
                tmp_event=tmp_event.replace(f'##ans_{i+1}##',tmp_answer_list[i])
            self.clerk.add_event(tmp_event)
            self.mod_ans = self.clerk.ask(700)
            self.clerk_results=self.mod_ans
            self.clerk.add_memory(self.mod_ans)

            tmp_event=self.config['moderator_prompt'].replace('##round##', self.round_dct(round+2))
            # tmp_event=tmp_event.replace('##format##','The previous sorting results are as follows. ##previous_result##. You need to merge the rebate results of the current round into this JSON file, ensuring that the \"Reason\" for the same \"Answer\" are summarized and merged together For new answers, please add them into JSON as follows: \"AnswerXX: xxx\": \"[xxx,xxx,..]\".')
            # tmp_event=tmp_event.replace('##previous_result##',self.clerk_results)
            for i in range(self.num_member):
                tmp_event=tmp_event.replace(f'##ans_{i+1}##',tmp_answer_list[i])
            self.moderator.add_event(tmp_event)
            self.mod_ans = self.moderator.ask(500)
            self.moderator.add_memory(self.mod_ans)
            try:
                self.mod_ans = eval(self.mod_ans)       
            except:
                print('Format error!')
                # self.mod_ans={'Reason':self.mod_ans}
            self.reason_list.append(self.mod_ans)
        # if self.mod_ans["Answer"] != '':
        #     self.config.update(self.mod_ans)
        #     self.config['success'] = True

        # # ultimate deadly technique.
        # else:
        judge_player = self.judge_player
        last_ans_list=[mem.memory_lst[-1]['content'] for mem in self.member]

        judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

        # extract answer candidates
        tmp_event=self.config['judge_prompt_last1']
        # tmp_event=tmp_event.replace('##iterate_add##',self.clerk_results)
        tmp=''
        for i in range(self.num_member):
            tmp+=f'Member {i+1} side provides the answer: {last_ans_list[i]}\n\n'
        tmp_event=tmp_event.replace('##iterate_add##',tmp)
        # for i in range(self.max_round): #TODO: add reasons here
        #     if not isinstance(self.reason_list[i],dict):
        #         tmp_event=tmp_event.replace(f'##ans0_{i+1}##', '')
        #         continue
        #     tmp_event=tmp_event.replace(f'##ans0_{i+1}##',str(self.reason_list[i]))
        judge_player.add_event(tmp_event)
        ans = judge_player.ask(500)
        self.debate_answer=ans
        judge_player.add_memory(ans)

        # select one from the candidates
        tmp_event=self.config['judge_prompt_last2']
        tmp_event=tmp_event.replace('##record##',self.clerk_results)
        # for i in range(self.max_round): #TODO: add reasons here
        #     if not isinstance(self.reason_list[i],dict):
        #         tmp_event=tmp_event.replace(f'##ans0_{i+1}##', '')
        #         continue
        #     tmp_event=tmp_event.replace(f'##ans0_{i+1}##',str(self.reason_list[i]))
        judge_player.add_event(tmp_event)
        ans = judge_player.ask(500)
        judge_player.add_memory(ans)
        
        try:
            ans = eval(ans)
        except:
            ans={'Reason':ans,'Answer':''}
        
        if not isinstance(ans,dict) or 'Reason' not in ans.keys():
            ans={'Reason':str(ans),'Answer':''}
        
        if ans["Answer"] != '':
            self.config['success'] = True
            self.debate_answer=ans
            # save file
        self.config.update(ans)
        self.players.append(judge_player)

        self.save_answser(save_dir=save_dir)

    def init_jury(self):
        JURY_LIST=[]
        for j in range(self.num_jury):
            JURY_LIST.append(f'Jury_{j+1}')
        if self.model_misc==None:
            if self.model_name=='llama':
                model,tokenizer=load_llama_model(model_name='7b',device=self.gpu_list[0])
                model_misc=(model,tokenizer)
            elif self.model_name=='vicuna':
                model,tokenizer=load_llama_model(model_name='vicuna',device=self.gpu_list[0])
                model_misc=(model,tokenizer)
            elif self.model_name=='mistral':
                model,tokenizer=load_llama_model(model_name='mistral',device=self.gpu_list[0])
                model_misc=(model,tokenizer)
            else:
                model_misc=self.model_misc
        else:
            model_misc=self.model_misc
        self.jury_members = [
        DebatePlayer(model_name=self.model_name,
        name=JURY_LIST[n],
        temperature=self.temperature,
        openai_api_key=self.openai_api_key,
        sleep_time=self.sleep_time,
        gpu=self.gpu_list[0],model_misc=model_misc) for n in range(len(JURY_LIST))
        ]

        for i in range(self.num_jury):
            new_prompt=get_role_prompt(self.config['jury_meta_prompt'],self.jury_list[i])
            self.jury_members[i].set_meta_prompt(new_prompt)

    def jury_evaluate(self,save_dir):
        # tmp_event=self.config['jury_prompt_1'].replace('##finalanswer##', self.config["Answer"])
        # reason_string=' '.join([str(i)+'.'+self.config["Reason"][i] for i in range(len(self.config["Reason"]))])
        # tmp_event=tmp_event.replace('##reasons##', reason_string)
        if self.only_jury:
            previous_result_path=os.path.join(save_dir,'result-0.pkl')
            if not os.path.exists(previous_result_path):
                print(f'No Valid result in {save_dir}')
                return 0
            with open(previous_result_path, 'rb') as f:#input,bug type,params
                result_dict = pickle.load(f)
            self.config["Answer"]=result_dict["debate_answer"]
            self.config["Reason"]=result_dict["Reason"]
            self.clerk_results=result_dict["All_Reasons"]
        tmp_event=self.config['jury_prompt_2'].replace('##finalanswer##', self.config["Answer"])
        reason_string=str(self.clerk_results)
        tmp_event=tmp_event.replace('##reasons##', reason_string)
        jury_result=[]
        for i in range(self.num_jury):
            self.jury_members[i].add_event(tmp_event)# argue the last one's point
            ans=self.jury_members[i].ask(500)
            try:
                ans = eval(ans)
            except:
                if 'True' in ans:
                    ans={'Add':'','Judge':'True'}
                else:
                    ans={'Add':'[]','Judge':'False'}
            self.jury_members[i].add_memory(ans)
            jury_result.append(ans)
        self.update_debate_result(jury_result)
        self.save_answser(save_dir=save_dir,misc='-jury')
    
    def jury_evaluate_single(self,save_path):

        if not os.path.exists(save_path):
            print(f'No Valid result in {save_path}')
            return 0
        with open(save_path, 'rb') as f:#input,bug type,params
            result_dict = pickle.load(f)
        try:
            tmp_result=get_total_reason_dict(result_dict['final_result'])
            self.config["Answer"]=tmp_result["Answer"]
            self.config["Reason"]=tmp_result["Reason"]
        except:
            self.config["Answer"]=''
            self.config["Reason"]=result_dict['final_result']
        tmp_event=self.config['jury_prompt_2'].replace('##finalanswer##', self.config["Answer"])
        reason_string=str(self.config["Reason"])
        tmp_event=tmp_event.replace('##reasons##', reason_string)
        jury_result=[]
        for i in range(self.num_jury):
            self.jury_members[i].add_event(tmp_event)# argue the last one's point
            ans=self.jury_members[i].ask(500)
            try:
                ans = eval(ans)
            except:
                if 'True' in ans:
                    ans={'Add':'','Judge':'True'}
                else:
                    ans={'Add':'[]','Judge':'False'}
            self.jury_members[i].add_memory(ans)
            jury_result.append(ans)
        self.update_debate_result(jury_result)

        tmp_save_path=save_path.replace('.pkl','-jury.pkl')
        result_dict["jury_memory_list"]=[[mem.name,mem.memory_lst] for mem in self.jury_members]
        result_dict["jury_answer"]=self.config['JuryAnswer']
        result_dict["jury_count"]=self.config['JuryNumber']
        result_dict['Reason']=self.config['Reason']
        with open(tmp_save_path, 'wb') as f:
            pickle.dump(result_dict, f)

        
    
    def update_debate_result(self,jury_result,threshold=0.83):
        count=0
        for result in jury_result:

            if isinstance(result,bool):
                if result:
                    count+=1
            elif not isinstance(result,dict):
                count+=1
            elif result['Judge']!='False':
                count+=1
            else:
                try:
                    self.config['Reason']+=result['Add']
                except:
                    pass
        self.config['JuryNumber']=count
        if count/self.num_jury>threshold:
            self.config['JuryAnswer']=True
            print('Success! Valid Result!')
        else:
            self.config['JuryAnswer']=False
            print('Debate Result Need to be Updated!')
