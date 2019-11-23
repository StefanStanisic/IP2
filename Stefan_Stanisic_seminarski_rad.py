import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
import time
from joblib import dump, load
import os


def fit_model(model, x_train, x_test, y_train, y_test, mode, method):
    if mode == 'save':
        model.fit(x_train, y_train.ravel())
        dump(model, os.path.join('models', method + '.joblib'), compress = 3)
    print(f"Rezultat trening skupa: {model.score(x_train, y_train):.3f}")
    print(f"Rezultat test skupa: {model.score(x_test, y_test):.3f}")

def prediction(model, x_train, x_test, y_train, y_test):
    y_train_predicted = model.predict(x_train)
    y_test_predicted = model.predict(x_test)
    print("Matrica kofuzije trening skupa:\n" + str(confusion_matrix(y_train, y_train_predicted)))
    print("Matrica kofuzije test skupa:\n" + str(confusion_matrix(y_test, y_test_predicted)))

# Klasifikacija pomocu K najblizih suseda, rtype predstavlja tip podatka koji cemo klasifikovati
# 'gen' za klasifikaciju gena i 'out' za klasifikaciju elemenata van granice
def knn(x_train, x_test, y_train, y_test, non, mode, rtype):
    print("-----K najblizih suseda:-----")
    start = time.time()
    # if mode == 'load':
    #     model = load(os.path.join('models', 'knn_' + rtype + '.joblib'))
    # else:
    model = KNeighborsClassifier(n_neighbors=non, weights='uniform')
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'knn_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Decision tree classifier
def dtc(x_train, x_test, y_train, y_test, mode, rtype, criteria, depth):
    print("-----Drvo odlucivanja:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'dtc_' + rtype + '.joblib'))
    else:
        model = DecisionTreeClassifier(criterion=criteria, max_depth=depth)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'dtc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Support vector machine
def svm(x_train, x_test, y_train, y_test, kernel, mode, rtype):
    print("-----SVM:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'svm_' + rtype + '.joblib'))
    else:
        model = SVC(kernel=kernel, degree=2, gamma='scale')
    # noinspection PyTypeChecker
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'svc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Random forest classifier
def rfc(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Nasumicna suma:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'rfc_' + rtype + '.joblib'))
    else:
        model = RandomForestClassifier(n_estimators=n_est, max_depth=5, criterion='gini')
    # noinspection PyTypeChecker
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'rfc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Bagging
def bag(x_train, x_test, y_train, y_test, n_est, mode, rtype, model):
    print("-----Bagging:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'bagging_' + rtype + '.joblib'))
    else:
        model = BaggingClassifier(base_estimator=model, n_estimators=n_est)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'bagging_' + n_est + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Boosting
def boost(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Boosting:-----")
    start = time.time()
    model = DecisionTreeClassifier(max_depth=5)
    if mode == 'load':
        model = load(os.path.join('models', 'boosting_' + rtype + '.joblib'))
    else:
        model = AdaBoostClassifier(base_estimator=model, n_estimators=n_est)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'boosting_' + n_est + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Voting classifier
def vot(x_train, x_test, y_train, y_test, mode, rtype):
    print("-----Voting:-----")
    start = time.time()
    est_model1 = RandomForestClassifier(n_estimators=100, max_depth=5)
    est_model2 = SVC(kernel='linear', gamma='scale')
    est_model3 = DecisionTreeClassifier(max_depth=5)
    if mode == 'load':
        model = load(os.path.join('models', 'voting_' + rtype + '.joblib'))
    else:
        model = VotingClassifier(estimators=[('dtc', est_model1), ('svc', est_model2),
                                             ('tree', est_model3)])

    est_model1.fit(x_train, y_train.ravel())
    est_model2.fit(x_train, y_train.ravel())
    est_model3.fit(x_train, y_train.ravel())

    fit_model(model, x_train, x_test, y_train, y_test, mode, 'voting_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


def main():
    # Ucitavanje podataka
    first_file = '087_CD4+_Helper_T_Cells_csv.csv'
    second_file = '088_CD4+CD25+_Regulatory_T_Cells_csv.csv'

    df_first = pd.read_csv(first_file)
    df_second = pd.read_csv(second_file)
    """
    ************************
       Pretprocesiranje
    ************************
    """

    # dfft -> data frame first transposed
    # dfst -> data frame second transposed
    # dfft i dfst sadrze transponovane podatke iz tablea df_first i df_second
    df1 = df_first.T
    df2 = df_second.T

    # Dodavanje kolone na kraju kao oznaka klase
    df1['class'] = 1
    df2['class'] = 2

    print("Dimenzija podataka prve datoteke: " + str(df1.shape))
    print("Dimenzija podataka druge datoteke: " + str(df2.shape) + '\n')

    # Spajanje podataka u jednu matricu, koju cemo nadalje koristiti u toku pretprocesiranja,
    # a zatim i klasifikacije
    df1.drop(df1.index[:1], inplace = True)
    df2.drop(df2.index[:1], inplace = True)

    # Spajaju se
    df = pd.concat([df1,df2]).sort_index(kind = 'merge')

    # Izbacivanje 0
    df = df.loc[:, (df != 0).any(axis = 0)]

    print('Dimenzije nakon ciscenja: '  + str(df.shape))

    """
    ************************
        Klasifikacija
    ************************
    """

    # X predstavlja podatke za klasifikaciju
    x = df.values[:, :-1]

    # Y predstavlja klase kojima podaci pripadaju
    y = df.values[:, -1:]
    y = y.astype('int')


    # Podela podataka na trening i test skup, odnos 70/30
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # ***** Jednostavne metode *****

    # K najblizih suseda
    """
    knn(x_train, x_test, y_train, y_test, non=3, mode='save', rtype='gen')
    knn(x_train, x_test, y_train, y_test, non=5, mode='save', rtype='gen')
    knn(x_train, x_test, y_train, y_test, non=10, mode='save', rtype='gen')
    """

    # Drvo odlucivanja
    """
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='gini', depth=None)
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='entropy', depth=None)
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='gini', depth=5)
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='entropy', depth=5)
    """

    # Masine sa potpornim vektorima

    """
    svm(x_train, x_test, y_train, y_test, kernel='rbf', mode='save', rtype='gen')
    svm(x_train, x_test, y_train, y_test, kernel='linear', mode='save', rtype='gen')
    svm(x_train, x_test, y_train, y_test, kernel='poly', mode='save', rtype='gen')
    """

    # ***** Ansambl tehnike *****

    # Nasumicna suma
    """
    rfc(x_train, x_test, y_train, y_test, n_est=10, mode='save', rtype='gen')
    rfc(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen')
    rfc(x_train, x_test, y_train, y_test, n_est=100, mode='save', rtype='gen')
    """

    # Pakovanje
    """
    bag(x_train, x_test, y_train, y_test, n_est=10, mode='save', rtype='gen',
       model = DecisionTreeClassifier(max_depth=5))
    bag(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen',
       model = DecisionTreeClassifier(max_depth=5))
    bag(x_train, x_test, y_train, y_test, n_est=5, mode='save', rtype='gen',
       model = SVC(kernel='linear', gamma='scale'))
    bag(x_train, x_test, y_train, y_test, n_est=20, mode='save', rtype='gen',
       model = SVC(kernel='linear', gamma='scale')) Ovaj algoritam nije izvrsen jer mu je potrebno previse vremena zbog velicine podataka. 
       Pokusano je njeovo izvrsavanje ali je trajalo vise od 12 sati pa je prekinuto kako bi bili izvrseni drugi algoritmi
    """

    # Pojacavanje
    """
    boost(x_train, x_test, y_train, y_test, n_est=10, mode='save', rtype='gen')
    boost(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen')
    boost(x_train, x_test, y_train, y_test, n_est=100, mode='save', rtype='gen')
    """

    # Glasanje
    
    vot(x_train, x_test, y_train, y_test, mode='save', rtype='gen')
    

if __name__ == '__main__':
    main()
