import numpy as np 
import pandas as pd
from sklearn.utils import shuffle


class Data_Preprocessing:
    def __init__(self,):
        Paths = {
            'Data': './../../Data/',
            'Chia_w_scope': './../../Data/chia_with_scope/',
            'Chia_wo_scope': './../../Data/chia_without_scope/',
        }
        self.path = Paths


    def eval_dataframe(self, df):

        df['Text'] = df['Text'].apply(eval)
        df['Group_Entities'] = df['Group_Entities'].apply(eval)
        df['Relations'] = df['Relations'].apply(eval)
        df['Tokens'] = df['Tokens'].apply(eval)
        df['Entities'] = df['Entities'].apply(eval)

        return df

    def get_tags_and_tokens(self, df):
        
        globalTags = []
        globalTokens = []
        prefixes = ['B-', 'I-']

        for i, f in df.iterrows():
            entities_in_file = f['Entities']
            tokens_in_file = f['Tokens']
            for ent in entities_in_file[0]:
                entity_arr = []
                for e in ent:
                    if (any(e.startswith(x) for x in prefixes)) or (e=='O'): entity_arr.append(e)
                    else: 
                        new_e = 'B-'+ e
                        entity_arr.append(new_e)                  
                globalTags.append(entity_arr)
            for t in tokens_in_file:
                globalTokens.append(t)
            assert len(globalTokens[-1]) == len(globalTags[-1])
        assert len(globalTokens) == len(globalTags) 

        return globalTokens, globalTags

    
    def create_dataframe(self, globalTokens, globalTags):
        ner_df = pd.DataFrame()
        ner_df['Tags'] = pd.Series(globalTags)
        ner_df['Sentence'] = pd.Series(globalTokens)
        
        return ner_df

    def get_entities_encoding_and_values(self, ner_df):
        entity_count = {}
        for i, f in ner_df.iterrows():
            tags = f['Tags']
            for t in tags:
                try:
                    entity_count[t] += 1
                except:
                    entity_count[t] = 1

        entities = list(entity_count.keys())
        entities.append('PAD')

        tag_idx = {t: i for i, t in enumerate(entities)}
        tag_values = list(tag_idx.keys())

        return tag_idx, tag_values
        

    def run(self,):
        df = pd.read_csv(self.path['Data'] + 'Chia_w_scope_data.csv')
        df = df.drop(columns=df.columns[0], axis=1)

        df = self.eval_dataframe(df)
        globalTokens, globalTags = self.get_tags_and_tokens(df)
        ner_df = self.create_dataframe(globalTokens, globalTags)

        # # Shuffling the dataframe
        # shuffled = shuffle(ner_df)
        # ner_df = shuffled.reset_index(drop=True)

        tag_idx, tag_values = self.get_entities_encoding_and_values(ner_df)

        return globalTokens, globalTags, tag_idx, tag_values, ner_df
        

    def main(self,):
        tokens, tags, tag_idx, tag_values, ner_df = self.run()
        return tokens, tags, tag_idx, tag_values

if __name__ == '__main__':
    dataPreprocessing = Data_Preprocessing()
    tokens, tags, tag_idx, tag_values = dataPreprocessing.main()
    print(tag_idx)
    print('-'*20)
    print(tag_values)
