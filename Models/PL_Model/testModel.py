import pandas as pd
from tqdm import tqdm

from sklearn.utils import shuffle

from inferencerClass import Inferencer


Paths = {
    'Data': './../Data/',
    'Nostos_Test' : './../../Data/PreparedText2SQL/test.csv'
}

df = pd.read_csv(Paths['Nostos_Test'])
df_test = shuffle(df)
model_path = './../Model_Files/sql_gen_model_checkpoint.ckpt' 
inferencer = Inferencer(model_path)

count, exact_match = 0, 0
# for i, f in tqdm(df_test.iterrows(), total=df_test.shape[0]):
for i, f in df_test.iterrows():
    Query = f.loc['Query_Generated']
    INPUT = f.loc['Folded_Question']
    output = inferencer(INPUT)

    count += 1
    if output == Query:
        exact_match += 1

    if count % 5 == 0:
        print('-'*10 + ' Epoch {}'.format(count))
        print('Mathes: {} Total Count: {}'.format(exact_match, count))
        exact_match_acc = (exact_match/count)*100
        print('Exact Match Accuracy: {}%'.format(exact_match_acc))


