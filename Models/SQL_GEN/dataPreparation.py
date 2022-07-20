import re
import string
import pandas as pd

from sklearn.utils import shuffle

from tqdm import tqdm
from itertools import product


def getSentences(s):
    def getAlphabetsList():
        alphabet_string = string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        return alphabet_list

    def getCombinations(s, alphabet_list):
        options = re.findall(r'<SYN-ARG-(.*?)>',s)
        words = [w.split('/') for w in options]

        columnNumber = len(words)
        alphabet_list = alphabet_list[:columnNumber]

        df = pd.DataFrame(words)
        df = df.transpose()
        df.columns = alphabet_list

        cartesianList = [df[alphabet_list[i]] for i in range(columnNumber)]
        combinations  = list(product(*cartesianList))
        combinations = list(filter(lambda x: None not in x, combinations)) # Remove pair with None
        return combinations, columnNumber

    def getPermutedSentences(s, combinations, columnNumber):
        SENTENCES = []
        options = re.sub(r'<SYN-ARG-(.*?)>', '#######',s)
        hashes = re.findall(r'#######', options)
        assert len(hashes) == columnNumber
        for c in combinations:
            for num_cols in range(columnNumber):
                options = options.replace('#######', c[num_cols], 1)
            SENTENCES.append(options)
            options = re.sub(r'<SYN-ARG-(.*?)>', '#######',s)
        assert len(combinations) == len(SENTENCES)
        return SENTENCES

    alphabet_list = getAlphabetsList()
    combinations, columnNumber = getCombinations(s, alphabet_list)
    sentences = getPermutedSentences(s, combinations, columnNumber)
    return sentences

def getDataFrames(df_type):
    FOLDED, BASE, QUERY = [], [], []
    for i, f in tqdm(df_type.iterrows(), total=df_type.shape[0]):
        folded_qsts = df_type.loc[i]['folded_questions']
        base_qsts = df_type.loc[i]['base_question']
        query_gen = df_type.loc[i]['query_generated']
        sentences = getSentences(folded_qsts)
        FOLDED += sentences
        BASE += [base_qsts]*len(sentences)
        QUERY += [query_gen]*len(sentences)
        assert len(FOLDED) == len(BASE) == len(QUERY)
    
    df = pd.DataFrame()
    df['Base_Question'] = pd.Series(BASE)
    df['Folded_Question'] = pd.Series(FOLDED)
    df['Query_Generated'] = pd.Series(QUERY)
    
    return df

def prepare_data():
    Paths = {
        'Data': './../../Data/',
        'Train': './../../Data/Text2SqlData/train.csv',
        'Validation': './../../Data/Text2SqlData/validation.csv',
        'Test': './../../Data/Text2SqlData/test.csv',
        'New_Train': './../../Data/PreparedText2SQL/train.csv',
        'New_Validation': './../../Data/PreparedText2SQL/validation.csv',
        'New_Test': './../../Data/PreparedText2SQL/test.csv',
    }

    df_train = pd.read_csv(Paths['Train'])
    df_val = pd.read_csv(Paths['Validation'])
    df_test = pd.read_csv(Paths['Test'])

    df_train.rename(columns = {'query':'query_generated'}, inplace = True)
    df_val.rename(columns = {'query':'query_generated'}, inplace = True)
    df_test.rename(columns = {'query':'query_generated'}, inplace = True)

    dataframes = [df_train, df_val, df_test]
    names = ['New_Train', 'New_Validation', 'New_Test']

    for i, d in enumerate(dataframes):
        print('Preparing {}...\n'.format(names[i]))
        f = getDataFrames(d)
        f.to_csv(Paths[names[i]])


if __name__ == '__main__':
    prepare_data()