from keras_preprocessing.sequence import pad_sequences


class Data_Processing:
    def __init__(self, sentences, text_labels, tokenizer, tag_idx, HYPER_PARAMETERS):
        self.sentences = sentences
        self.text_labels = text_labels
        self.tokenizer = tokenizer
        self.tag_idx = tag_idx
        self.HYPER_PARAMETERS = HYPER_PARAMETERS

        # List of tuples of tokenized sentences
        self.tokenized_texts_and_labels = [self.tokenize_preserve(sent,labs) for sent,labs in zip(self.sentences,self.text_labels)]
        # Seperates tokenized pairs into labels and tokens
        self.tokenized_text = [token_label_pair[0] for token_label_pair in self.tokenized_texts_and_labels]
        self.labels = [token_label_pair[1] for token_label_pair in self.tokenized_texts_and_labels] 

    def tokenize_preserve(self,sent,labs):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sent,labs):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label]*n_subwords)

        return tokenized_sentence, labels

    def getProcessedData(self,):
        input_ids = pad_sequences(
                              [self.tokenizer.convert_tokens_to_ids(txt) for txt in self.tokenized_text], # converts tokens to ids
                             maxlen= self.HYPER_PARAMETERS['MAX_LEN'], dtype='long',value=0.0,
                             truncating='post',padding='post')
        tags = pad_sequences(
                        [[self.tag_idx.get(l)for l in lab]for lab in self.labels], # Gets corresponding tag_id
                        maxlen= self.HYPER_PARAMETERS['MAX_LEN'], dtype='long', value=self.tag_idx['PAD'],
                        truncating='post',padding='post')

        attention_masks = [[float(i !=0.0) for i in ii]for ii in input_ids] # Float(True) = 1.0 for attention for only non-padded inputs

        # self.printOutputs(input_ids, tags, attention_masks)

        return input_ids, tags, attention_masks

    def printOutputs(self,input_ids, tags, attention_masks):
        print('Inputs: {}'.format(input_ids[0]))
        print('Tags: {}'.format(tags[0]))
        print('Attention Mask: {}'.format(attention_masks[0]))
        print('Lengths Matching: {}, {}, {}'.format(len(input_ids[0]), len(tags[0]), len(attention_masks[0])))