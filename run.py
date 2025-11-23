from datasets import load_dataset
from tqdm.auto import tqdm
import time
import numpy as np
from utils import QueryModelError, QueryJudgeError, GetJudgePrompt, SaveJson, LoadJson

def run(benchmark, model, n = 200):
    if benchmark=='ifeval':
        from ifeval.utils import process_results
        bench_data = load_dataset("google/IFEval")['train']
    elif benchmark=='math':
        from mathbench.utils import doc_to_text
        bench_data = load_dataset("DigitalLearningGmbH/MATH-lighteval","default")
        bench_data = [d for d in bench_data['test'] if d['level']=='Level 5']
    elif benchmark=='musr':
        from musr.utils import doc_to_text
        bench_data = load_dataset("TAUR-Lab/MuSR")
        bench_data_aux = []
        for subject in ['murder_mysteries','object_placements','team_allocation']:
            bench_data_aux2 = list(bench_data[subject])
            for d in bench_data_aux2:
                d['subject'] = subject
            bench_data_aux += bench_data_aux2
        bench_data = bench_data_aux
        del(bench_data_aux)
        del(bench_data_aux2)
    elif benchmark=='gpqa':
        from gpqa.utils import process_docs,doc_to_text
        bench_data = load_dataset("Idavidrein/gpqa",'gpqa_main')['train']
        bench_data = list(process_docs(bench_data))
    elif benchmark=='bbh':
        from bbh.utils import doc_to_text
        bench_data = []
        subs = ['boolean_expressions', 'causal_judgement', 'date_understanding',
               'disambiguation_qa', 'formal_fallacies', 'geometric_shapes',
               'hyperbaton', 'logical_deduction_five_objects',
               'logical_deduction_seven_objects',
               'logical_deduction_three_objects', 'movie_recommendation',
               'navigate', 'object_counting', 'penguins_in_a_table',
               'reasoning_about_colored_objects', 'ruin_names',
               'salient_translation_error_detection', 'snarks',
               'sports_understanding', 'temporal_sequences',
               'tracking_shuffled_objects_five_objects',
               'tracking_shuffled_objects_seven_objects',
               'tracking_shuffled_objects_three_objects', 'web_of_lies']
        for subject in tqdm(subs,desc='loading bbh data'):
            bench_data_aux = list(load_dataset("SaylorTwift/bbh",subject)['test'])
            for d in bench_data_aux:
                d['subject'] = subject
            bench_data += bench_data_aux
        del(bench_data_aux)
    elif benchmark=='mmlu-pro':
        from mmlupro.utils import doc_to_text, doc_to_choice
        bench_data = list(load_dataset("TIGER-Lab/MMLU-Pro")['test'])
    else:
        raise(NotImplementedError)
        
    # ids to eval
    m = len(bench_data) 
    ids = np.unique(np.round(np.linspace(0,m-1,n)))
    #ids += 1 # +1 because we want to start from 1 (we are collecting for the second time, so want to minimize intersection with first run)
    ids = [int(x) for x in ids if int(x) <= m-1]

    for id in tqdm(ids):
        
        model_name = model.replace("/",".").replace("_",".")
        bench_name = benchmark.replace("/",".").replace("_",".")
        file_path = f"results/{bench_name}/{model_name}/{bench_name}_{model_name}_{id}.json"

        try:
            output = LoadJson(file_path)
            prompt = output['prompt']
            gold_response = output['gold_response']
            doc = output['doc']

            if output['model_response'] == "NA": #getting model resp if NA
                model_response = QueryModelError(prompt, model)
                output['model_response'] = model_response
                SaveJson(output,file_path) #we just re-save if changes are made
            else:
                model_response = output['model_response']
            if model_response != "NA":
                if output['scores'] == "NA": #getting scores if NA
                    if benchmark=='ifeval':
                        scores = process_results(doc, [model_response])
                        judge_prompt="NA"
                    else:
                        judge_prompt=GetJudgePrompt(prompt, model_response, gold_response)
                        scores = QueryJudgeError(judge_prompt)
                    output['scores'] = scores
                    output['judge_prompt'] = judge_prompt
                    SaveJson(output,file_path) #we just re-save if changes are made
        except:    
            ### Getting prompt, gold resp, subject, and id bench
            doc = bench_data[id]
            if benchmark=='ifeval':
                prompt = doc['prompt']
                gold_response = ''
                subject = 'ifeval'
                id_bench = doc['key']
            elif benchmark=='math':
                prompt = doc_to_text(doc)
                gold_response = doc['solution']
                subject = doc['type']+"_"+doc['level'].replace(" ","-")
                id_bench = ''
            elif benchmark=='musr':
                prompt = doc_to_text(doc)
                gold_response = doc['answer_choice']
                subject = doc['subject']
                id_bench = ''
            elif benchmark=='gpqa':
                prompt = doc_to_text(doc)
                gold_response = doc['answer']
                subject = 'main'
                id_bench = ''
            elif benchmark=='bbh':
                prompt = doc_to_text(doc)
                gold_response = doc['target']
                subject = doc['subject']
                id_bench = ''
            elif benchmark=='mmlu-pro':
                prompt = doc_to_text(doc)
                gold_response = doc['answer']
                subject = doc['category']
                id_bench = doc['question_id']
                
            ### Prompting model and evaluating
            model_response = QueryModelError(prompt, model)
            if model_response == "NA":
                scores = "NA"
                judge_prompt = "NA"
            else:
                if benchmark=='ifeval':
                    scores = process_results(doc, [model_response])
                    judge_prompt="NA"
                else:
                    judge_prompt=GetJudgePrompt(prompt, model_response, gold_response)
                    scores = QueryJudgeError(judge_prompt)
                    
            ### Output
            output={'model':model,
                    'benchmark':benchmark,
                    'id':id,
                    'id_bench':id_bench,
                    'prompt':prompt,
                    'model_response':model_response,
                    'gold_response':gold_response,
                    'subject':subject,
                    'scores':scores,
                    'judge_prompt':judge_prompt,
                    'doc':doc}
            
            SaveJson(output,file_path)