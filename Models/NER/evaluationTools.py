from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import numpy as np
import more_itertools

def getPredictions(A, P):   
    def getEntityIndices(A):   
        entity_index = []
        for i, a in enumerate(A):
            if a.startswith('B-'):
                if i == len(A)-1:
                    entity_index.append((i,i,a[2:]))
                else:
                    for j, b in enumerate(A, start=i+1):
                        if A[j].startswith('I-'):
                            if j == (len(A)-1):
                                entity_index.append((i,j,a[2:]))
                                break
                            pass
                        else:
                            entity_index.append((i,j-1,a[2:]))
                            break
            elif a == 'O':
                if i == len(A)-1:
                    entity_index.append((i,i,a))
                else:
                    for j, b in enumerate(A, start=i+1):
                        if A[j] == 'O':
                            if j == (len(A)-1):
                                entity_index.append((i,j,a))
                                break
                            pass
                        else:
                            entity_index.append((i,j-1,a))
                            break

        return entity_index

    def removeMultipleOs(C):
        I = [i for i, c in enumerate(C) if c[2] == 'O']        
        grplist = [list(group) for group in more_itertools.consecutive_groups(I)]
        indexList = [grp[0] for grp in grplist]
        removeList = []
        for i, c in enumerate(C):
            if (c[2] == 'O') and (i not in indexList):
                removeList.append(i)
        C = [i for j, i in enumerate(C) if j not in removeList]
        return C

    def getTrueLabels(entity_index):
        true_label_entities = []
        for entities in entity_index:
            true_label_entities.append(entities[2])
        return true_label_entities


    def getPredLabels(P, entity_index):
        prediction_entities = []
        for j, entities in enumerate(entity_index):
            boolFindOverlap = correctOverlap = False
            temp = 'Blah'
            for i in range(entities[0],entities[1]+1):
                if P[i].startswith('B-') or P[i].startswith('I-'):
                    boolFindOverlap = True 
                    temp = P[i][2:]
                    if P[i][2:] == entities[2]:
                        correctOverlap = True
                        prediction_entities.append(entities[2])
                        break
                else:
                    continue
            if boolFindOverlap == False and correctOverlap == False:
                prediction_entities.append('O')
            elif boolFindOverlap == True and correctOverlap == False:
                prediction_entities.append(temp)

            assert (j+1)==len(prediction_entities)

        return prediction_entities

    entity_index = getEntityIndices(A)
    entity_index = removeMultipleOs(entity_index)
    true_label_entities = getTrueLabels(entity_index)
    prediction_entities = getPredLabels(P, entity_index)
    
    assert len(true_label_entities) == len(prediction_entities)
    return true_label_entities, prediction_entities


def getReports(true_label_entities, prediction_entities):

    def classifcationReport(true_label_entities, prediction_entities):
        CR = classification_report(true_label_entities, prediction_entities)
        CR_Dict = classification_report(true_label_entities, prediction_entities, output_dict=True)
        return CR, CR_Dict
    
    def getConfusionMatrix(true_label_entities, prediction_entities, CR_Dict):
        labels_ = list(CR_Dict.keys())[:-3]
        confusionMatrix = confusion_matrix(true_label_entities, prediction_entities, labels=labels_)
        return confusionMatrix
    
    def getAccuracy(confusionMatrix, CR_Dict):
        labels = list(CR_Dict.keys())[:-3]
        acc = np.diag(confusionMatrix)/confusionMatrix.sum(1)
        
        return labels, acc
    
    CR, CR_Dict = classifcationReport(true_label_entities, prediction_entities)
    confusionMatrix = getConfusionMatrix(true_label_entities, prediction_entities, CR_Dict)
    labels, acc = getAccuracy(confusionMatrix, CR_Dict)

    return CR, labels, acc
    
def Eval(predictions, true_labels, tag_values):
    PREDICTIONS, TRUE_LABELS = [], []
    for pred, t_label in zip(predictions, true_labels):
        assert len(pred) == len(t_label)
        zipList = [(tag_values[p],tag_values[t])  for p, t in zip(pred, t_label) if tag_values[t]!= 'PAD']
        if len(zipList) == 0:
            continue
        else:
            pred, labels = zip(*zipList)
            pred, labels = list(pred), list(labels)
            assert len(pred) == len(labels)

            true_label_entities, prediction_entities = getPredictions(labels, pred)
            PREDICTIONS.extend(prediction_entities)
            TRUE_LABELS.extend(true_label_entities)
    
    assert len(PREDICTIONS) == len(TRUE_LABELS)
    CR, labels, acc= getReports(TRUE_LABELS, PREDICTIONS)
    print(CR)
    return CR, labels, acc
    

if __name__ == '__main__':
    pass