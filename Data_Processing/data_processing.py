import os
import re
import pandas as pd
from more_itertools import locate



Paths = {
    'Data': './../Data/',
    'Chia_w_scope': './../Data/chia_with_scope/',
    'Chia_wo_scope': './../Data/chia_without_scope/'
}


entity_types = ['Condition', 'Drug', 'Procedure', 'Measurement', 'Observation', 'Person', 'Device', \
    'Value', 'Temporal', 'Qualifier', 'Negation']

relation_type = ['OR', 'AND', 'Has_qualifier', 'Has_value', 'Has_negation', 'Has_temporal', 'Has_context']


class Data_Preparation:
    def __init__(self, path, relation_type, entity_types):
        self.path = path
        self.entity_types = entity_types
        self.relation_type = relation_type
        self.ignore_files = []
        self.input_files = []
        self.globalEntities = set()
        
        print('Data_Preparation Initialized...')
        
    ###################################################################### HELPER FUNCTIONS
        
    def checkEntityValue(self, e):
        """
        Helper Function for get_annotation_relations
        """
        if e.startswith('T'):
            return e.strip()
        else:
            return e.split(':')[-1].strip()
        
    
    def removePunctuation(self, word):
        """
        Helper Function for get_Entity_Tags
        """
        word = re.sub(r'^(\.|,|\(|\))', '', word)
        word = re.sub(r'(\.|,|\(|\))$', '', word)
        return word
    
    def cleanEntityTokens(self, txt):
        """
        Clean the tokens from / and - for words in entities
        """
        txt = txt.replace('/',' ')
        txt = txt.replace('-',' ')
        txt = txt.replace(';',' ')
        txt = re.sub(r' {2,}', ' ',txt)
        txt = txt.strip()
        
        return txt.split()
    
    def readTxt(self, txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            text_array = f.read()
        print(text_array)
        
    def readAnn(self, ann_file):
        with open(ann_file, "r", encoding="utf-8") as f:
            text_array = f.read()
        print(text_array)
        
    def printFiles(self, file_name, file):
        """
        Helper function to print files without manual viewing them
        """
        ann_file = f"{Paths[file_name]}{file}.ann"
        txt_file = f"{Paths[file_name]}{file}.txt"
        self.readAnn(ann_file)
        self.readTxt(txt_file)
    
    
    ######################################################################
    
    def getInputFiles(self, file_name):
        """
        file_name: 'Chia_w_scope' or 'Chia_wo_scope'
        """
        inputfiles = set()
        for f in os.listdir(Paths[file_name]):
            if f.endswith('.ann'):
                inputfiles.add(f.split('.')[0].split('_')[0])
        self.input_files = list(inputfiles)
        return inputfiles

    def ignoreFiles(self, text_array):
        match = re.findall(r'^( {1,}\n$|NA {0,}\n$)',text_array[0])
        if len(match):
            return True
        else:
            return False
    
    def get_annotation_entities(self, ann_file):
        """
        Get all entities which correspond to the entities mentioned in the entity types.
        """
        entities = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith('T'):
                    assert len(line.strip().split('\t')) == 3
                    entity_identity = line.strip().split('\t')[0]
                    entity_token = line.strip().split('\t')[-1]
                    
                    if ';' in line.strip().split('\t')[1]:
                        line = line.replace(';',' ')
                        term = line.strip().split('\t')[1].split()
                        term[-1] = str(max([int(v) for v in term[1:]]))
                        term[1] = str(min([int(v) for v in term[1:]]))
                    else:
                        term = line.strip().split('\t')[1].split()
                    self.globalEntities.add(term[0])
                    
                    if (self.entity_types != None) and (term[0] not in self.entity_types): continue
                    if int(term[-1]) <= int(term[1]):
                        raise RuntimeError('Starting and Ending Indices are off.')
                    entities.append((entity_identity, int(term[1]), int(term[-1]), term[0], entity_token))
                    
        return sorted(entities, key=lambda x: (x[2]))
    
    def remove_overlap_entities(self, sorted_entities):
        """
        If you want to get the largest overlap of entity so two words don't have different entities
        
        Here we just use the uncommented part to get the unique entities which we are considering.
        """
#         keep_entities = []
#         for idx, entity in enumerate(sorted_entities):
#             if idx == 0:
#                 keep_entities.append(entity)
#                 last_keep = entity
#                 continue
#             if entity[1] < last_keep[2]:
#                 if entity[2]-entity[1] > last_keep[2]-last_keep[1]:
#                     last_keep = entity
#                     keep_entities[-1] = last_keep
#             elif entity[1] == last_keep[2]:
#                 last_keep = (last_keep[1], entity[2], last_keep[-1])
#                 keep_entities[-1] = last_keep
#             else:
#                 last_keep = entity
#                 keep_entities.append(entity)

        keep_entities = sorted_entities

        uniqueEntity = []        
        for ent in keep_entities:
            uniqueEntity.append(ent[0])

        return keep_entities, uniqueEntity


    # https://datagy.io/python-list-find-all-index/
    def get_annotation_relations(self, ann_file, uniqueEntity):
        """
        Gives all relations corresponding to the relations mentioned in relations_type
        And make sure they are relations between entities that are mentioned in entity_types
        """
        relations = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith('R') or line.startswith('*'):
                    assert len(line.strip().split('\t')) == 2 # ['R1', 'Has_qualifier Arg1:T6 Arg2:T5']
                    if line.strip().split('\t')[1].split()[0] not in relation_type: continue

                    rel = line.strip().split('\t')[0]
                    rel_type = line.strip().split('\t')[1].split()[0]
                    entities = line.strip().split('\t')[1].split()[1:]
                    entities= [self.checkEntityValue(e) for e in entities]
                    entities = [e for e in entities if e in uniqueEntity]
                    entities = ' '.join(entities)
                    match = re.findall(r'^T[0-9]+ T[0-9]+$',entities)
                    if len(match):
                        pass
                    else:
                        continue
                    assert len(entities.split()) == 2
                    relations.append((rel, rel_type, entities))

        return relations
    
    
    def get_text(self, txt_file, file):
        """
        Get raw text
        """
        with open(txt_file, "r", encoding="utf-8") as f:
            text_array = f.readlines()
            if file in ['NCT02348918_exc', 'NCT02348918_inc', 'NCT01735955_exc']: # Inconsistent offsets (Line breaks)
                text = ' '.join([i.strip() for i in text_array])
            else:
                text = '  '.join([i.strip() for i in text_array])
        
        return text, text_array
    
    
    def get_text_array(self,text_array):
        """
        Get cleaned text in array form
        """
        
        globalText = []
        offset = 0
        for txt in text_array:
            textlen = len(txt)
            
            txt = txt.replace('/',' ')
            txt = txt.replace('-',' ')
            txt = txt.replace(';',' ')
            txt = re.sub(r' {2,}', ' ',txt)
            txt = txt.replace('.\n','')
            txt = txt.replace('\n', '')
            txt = txt.strip()

            globalText.append(([self.removePunctuation(w) for w in txt.split()], offset, offset + textlen)) 
            offset += textlen

        return globalText
    
    def get_NER_Tags(self, globalText, keep_entities):
        """
        Get NER Tags for each token in a sentence.
        Then also return a list of the entity identity e.g. T1, T2 etc.
        """
        WORDS, TAGS, IDENTITY = [], [], []
        offset = 10
        for text in globalText:
            words = text[0]
            tags = ['O']*len(words)
            entity_identity = ['X']*len(words)
            sent_indices = set()
            for k in keep_entities:
                # Using 10 as an offset for wrong offset entries in inc/exc files
                # Only look at entities if start and stop indices with an offset match
                if k[1] >= (text[1]-offset) and k[2] <= (text[2]+offset): 
                    clean_tokens = self.cleanEntityTokens(k[-1])
                    break_down = [self.removePunctuation(v) for v in clean_tokens] # Get all the tokens (punc removed) from entities
                    main_index = 0
                    label = ''
                    for i, w in enumerate(break_down): # Go over entity tokens
                        indices = list(locate(words, lambda x: x == w)) # See if tokens is in sentence

                        if i == 0:
                            try:
                                main_index = indices[0]
                                if len(break_down) > 1: label = 'B-'
                            except:
                                if len(w) <= offset:
                                    pass
                                else:
                                    pass
#                                     raise RuntimeError('Word Length greater than offset')
                        else:
                            label = 'I-'
                        indices= list(filter(lambda x: x >= main_index, indices))
                        indices= list(filter(lambda x: x not in sent_indices, indices))
                        if len(indices) != 0:
                            sent_indices.add(indices[0])
                            tags[indices[0]] = label + k[3]
                            entity_identity[indices[0]] = k[0]
            assert len(words) == len(tags) == len(entity_identity)
            WORDS.append(words)
            TAGS.append(tags) 
            IDENTITY.append(entity_identity)
        return WORDS, TAGS, IDENTITY
    
    def save_to_df(self, FILES, CRITERIA, TEXT, GROUP_ENTITIES, RELATIONS, TOKENS, ENTITIES):
        
        df = pd.DataFrame()

        df['File'] = pd.Series(FILES) 
        df['Criteria'] = pd.Series(CRITERIA) 
        df['Text'] = pd.Series(TEXT)
        df['Group_Entities'] = pd.Series(GROUP_ENTITIES)
        df['Relations'] = pd.Series(RELATIONS)
        df['Tokens'] = pd.Series(TOKENS)
        df['Entities'] = pd.Series(ENTITIES) # (Entity Name, Entity Identity)
        
        return df
    
    def run(self, file_name):
        """
        Run on all files
        """
        inputfiles = self.getInputFiles(file_name)
        
        FILES, CRITERIA, TEXT, GROUP_ENTITIES, RELATIONS, TOKENS, ENTITIES = [], [], [], [], [], [], []
        for infile in inputfiles:
            for t in ["inc", "exc"]:
                file = f"{infile}_{t}"
                ann_file = f"{Paths[file_name]}{file}.ann"
                txt_file = f"{Paths[file_name]}{file}.txt"
                
                text, text_array = self.get_text(txt_file, file)
                ignore = self.ignoreFiles(text_array)
                if ignore: 
                    self.ignore_files.append(file)
                    continue
                sorted_entities = self.get_annotation_entities(ann_file)
                entities, uniqueEntity = self.remove_overlap_entities(sorted_entities)
                relations = self.get_annotation_relations(ann_file, uniqueEntity)
                global_text_array = self.get_text_array(text_array)
                words, tags, entity_identity = self.get_NER_Tags(global_text_array, entities)           
                
                FILES.append(file)
                CRITERIA.append(t)
                TEXT.append(text_array)
                GROUP_ENTITIES.append(entities)
                RELATIONS.append(relations)
                TOKENS.append(words)
                ENTITIES.append((tags, entity_identity))
#             break
                           
        df = self.save_to_df(FILES, CRITERIA, TEXT, GROUP_ENTITIES, RELATIONS, TOKENS, ENTITIES)       
        return df
                
                
    def main(self,files_name):
        """
        Give list of files and from the files extract entities, tokens and relations and save in dataframe format
        """
        
        df = self.run(files_name)
        return df
      
if __name__ == '__main__':
    print('Preparing Data...')
    files = ['Chia_w_scope', 'Chia_wo_scope']
    dataPrep = Data_Preparation(Paths, relation_type, entity_types)
    df_w = dataPrep.main(files[0])
    df_w.to_csv(Paths['Data'] + 'Chia_w_scope_data.csv')
    print('Saved Chia_w_scope_data...')
    df_wo = dataPrep.main(files[1])
    df_wo.to_csv(Paths['Data'] + 'Chia_wo_scope_data.csv')
    print('Saved Chia_wo_scope_data...')
    print('Data Preparation Completed!')