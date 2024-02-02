import os
import pickle
import sys

def get_reason_list(reason_list_string):
    answer=''
    try:
        reason_dict=eval(reason_list_string)
    except:
        if reason_list_string[-1]=='.':
            reason_list_string=reason_list_string[:-1]
        if '{' in reason_list_string and '}' in reason_list_string:
            reason_list_string=reason_list_string[:reason_list_string.index('}')+1]

        if reason_list_string[-1]!='}':
            if reason_list_string[-1]==']':
                reason_list_string+='}'
            elif reason_list_string[-1]=='\"':
                reason_list_string+=']}'
            else:
                reason_list_string+='\"]}'
        try:
            reason_dict=eval(reason_list_string)
        except:
            reason_dict={}
    if isinstance(reason_dict,tuple):
        reason_dict=reason_dict[0]
    if isinstance(reason_dict,dict):
        if reason_dict!={}:
            try:
                reason_list=reason_dict['Reason']
            except:
                reason_list=[]
            try:
                answer=reason_dict['Answer']
            except:
                answer=''
        else:
            try:
                answer,tmp_reason=reason_list_string.split('\"Reason\":')
                answer=answer.split(':')[-1]
                reason_list=tmp_reason.split('\",')
            except ValueError as e:
                answer=reason_list_string.split('. ')[0]
                reason_list_string=reason_list_string.replace(answer,'')
                reason_list=reason_list_string.split('\n\n')
    else:
        print(1) 
    if not isinstance(reason_list, list):
        if '[' in reason_list and ']' in reason_list:
            reason_list=reason_list.split(', ')
        else:
            reason_list=reason_list.split('. ')
    for i in range(len(reason_list)):
        if isinstance(reason_list[i],set):
            reason_list[i]=str(reason_list[i])
    if len(reason_list)>20:
        print(1)
    return answer,reason_list

def get_total_reason_list(reason_list_string):
    output_list=[]
    try:
        reason_dict=eval(reason_list_string)
    except:
        if reason_list_string[-1]=='.':
            reason_list_string=reason_list_string[:-1]
        if '{' in reason_list_string and '}' in reason_list_string:
            reason_list_string=reason_list_string[:reason_list_string.index('}')+1]

        if reason_list_string[-1]!='}':
            if reason_list_string[-1]==']':
                reason_list_string+='}'
            elif reason_list_string[-1]=='\"':
                reason_list_string+=']}'
            else:
                reason_list_string+='\"]}'
        try:
            reason_dict=eval(reason_list_string)
        except:
            reason_dict={}
    if isinstance(reason_dict,tuple):
        reason_dict=reason_dict[0]
    
    if isinstance(reason_dict,set):
        output_list=reason_list_string.split('\",')
    else:
        for key in reason_dict.keys():
            if isinstance(reason_dict[key],str):
                reason_dict[key]=reason_dict[key].split(', ')
            output_list+=reason_dict[key]
        if reason_dict=={}:
            output_list=reason_list_string.split('\",')
    if not isinstance(output_list, list):
        print(11)
    if len(output_list)>50:
        print(1)
    return output_list

def get_jury_count(jury_result):
    count=0
    for result in jury_result:
        try:
            content=eval(result[-1][2]['content'])
            if content['Judge']!='False':
                count+=1
        except:
            if "\'True\'" in result[-1][2]['content']:
                count+=1
    return count