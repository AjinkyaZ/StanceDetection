import sys
import numpy as np

from models import FakeNet
from feature_engineering import refuting_features, polarity_features, hand_features, \
    bow_averaged_vectors, bow_count_vectors, gen_or_load_feats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from figures import graph_losses
from utils.system import parse_params, check_version
from tqdm import tqdm


def generate_features(stances,dataset,name):
    print("Generating Features for :", name)
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_bowv = np.array(gen_or_load_feats(bow_averaged_vectors, h, b, "features/bowvec_200dnorm."+name+".npy"))
    X_bowc = np.array(gen_or_load_feats(bow_count_vectors, h, b, "features/bowcount_1000."+name+".npy"))
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_bowv, X_bowc]
    print("... Done. Features :", X.shape[1])
    return X, y


if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet(path='../data/train')
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test", path='../data/competition_test')
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    print("competition test", X_competition.shape)
    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout")
    print("holdout", X_holdout.shape)
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None
    best_losses = []

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))
        print("fold train", X_train.shape)
        X_test = Xs[fold]
        y_test = ys[fold]
        print("fold test", X_test.shape)
        #clf = LogisticRegression()
        #clf = SVC(kernel='rbf')
        wts_fold = list(map(lambda x: 1/float(x), sorted(np.unique(y_train, return_counts=True), key=lambda x: x[0])[1]))
        #print(wts_fold)
        #clf = LogisticRegression()
        clf = FakeNet()
        if fold==0:
            print(clf)
        #clf.fit(X_train, y_train) #non-neural models
        losses = []
        losses = clf.fit(X_train, y_train, wts_fold)
        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf
            best_losses = losses


    graph_losses(losses, 'MLP')
    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")
    print("Macro F1 for Dev Set:", f1_score(actual, predicted, average='macro'))
    #report_score(actual, actual)
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]
    print("Macro F1 for Test Set:", f1_score(actual, predicted, average='macro'))

    print("Scores on the test set")
    report_score(actual, predicted)