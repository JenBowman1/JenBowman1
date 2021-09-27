from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('homepage.html')

@app.route("/house")
def homes():
    return render_template('house.html')

@app.route("/houseprediction", methods=["POST"])
def make_prediction():
    def importing():
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        with open('columns.pickle', 'rb') as f:
            cols = pickle.load(f)
        with open('transformer.pickle', 'rb') as f:
            transformer = pickle.load(f)
        with open('skewed_cols.pickle', 'rb') as f:
            skewness = pickle.load(f)
        with open('z.pickle', 'rb') as f:
            medians = pickle.load(f)
        return model, cols, transformer, skewness, medians
    model, cols, transformer, skewness, medians = importing()
    data = pd.DataFrame([request.form])
    columns = list(set(medians.columns)-set(data.columns))
    medians = medians[columns]
    medians = medians[medians.index==0]
    data = pd.concat([data, medians], axis=1)
    data = data.apply(pd.to_numeric, errors='ignore')

    def clean_data(data):
        data['Area'] = data['LotArea'] * data['LotFrontage']
        data['Total_Bathrooms'] = data['FullBath'] + \
            (0.5 * data['HalfBath']) + data['BsmtFullBath'] + \
            (0.5 * data['BsmtHalfBath'])
        data['MedNhbdArea'] = data.groupby(
            'Neighborhood')['GrLivArea'].transform('median')
        data['IsAbvGr'] = data[['MedNhbdArea', 'GrLivArea']].apply(
            lambda x: 'yes' if x['GrLivArea'] > x['MedNhbdArea'] else 'no', axis=1)
        data['MedNhbdBY'] = data.groupby('Neighborhood')[
            'BackyardSF'].transform('median')
        data['IsBigBY'] = data[['MedNhbdBY', 'BackyardSF']].apply(
            lambda x: 'yes' if x['BackyardSF'] > x['MedNhbdBY'] else 'no', axis=1)
        data['ModernHouse'] = data[['Utilities', 'HeatingQC', 'CentralAir', 'Electrical']].apply(lambda x: 'yes' if x['Utilities'] == 'AllPub' and (
            x['HeatingQC'] == 'Ex' or x['HeatingQC'] == 'Gd') and x['CentralAir'] == 'Y' and x['Electrical'] == 'SBrkr' else 'no', axis=1)
        data['pca_1'] = data.GrLivArea + data.TotalBsmtSF
        data['pca_2'] = data.YearRemodAdd * data.TotalBsmtSF
        data['AgeofHouse'] = 2010 - data['YearBuilt']
        data['hasreglot'] = data['LotShape'].apply(lambda x: 1 if x == 'Reg' else 0)
        data['haslevellot'] = data['LandContour'].apply(
            lambda x: 1 if x == 'Lvl' else 0)
        data['hasgentleslope'] = data['LandSlope'].apply(
            lambda x: 1 if x == 'Gtl' else 0)
        data['hasdetachedGr'] = data['GarageType'].apply(
            lambda x: 1 if x == 'Detchd' else 0)
        data['hasshed'] = data['MiscFeature'].apply(
            lambda x: 1 if x == 'Shed' else 0)
        data['wasremodeled'] = data[['YearRemodAdd', 'YearBuilt']].apply(
            lambda x: 1 if x['YearRemodAdd'] != x['YearBuilt'] else 0, axis=1)
        data['isnewhouse'] = data[['YearBuilt', 'YrSold']].apply(
            lambda x: 1 if x['YearBuilt'] == x['YrSold'] else 0, axis=1)
        data['isgoodNbhd'] = data['Neighborhood'].apply(
            lambda x: 1 if x == 'NridgHt' or x == 'Crawfor' or x == 'StoneBr' or x == 'Somerst' or x == 'NoRidge' else 0)
        return data
    X = clean_data(data)

    def fixing_skew(data, skewness):
        trans_cats = list(skewness.index)
        for x in trans_cats:
            data[x] = np.log1p(data[x])
        return data
    X = fixing_skew(X, skewness)

    def scale(data, transformer):
        z = ['MedNhbdBY', 'pca_1', 'pca_2', 'MedNhbdArea', 'BackyardSF', 'PorchSF', '1stFlrSF', '2ndFlrSF', 'AllSF', 'Area', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'TotalBsmtSF']
        q = ['hasreglot', 'haslevellot', 'hasgentleslope', 'hasdetachedGr', 'hasshed', 'wasremodeled', 'isnewhouse', 'isgoodNbhd', 'AgeofHouse', 'IsBigBY', 'ModernHouse', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 'BsmtExposure', 'IsAbvGr', 'Total_Bathrooms', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'MiscVal', 'OverallCond', 'BsmtFinType2', 'SaleType', 'YrSold', 'MoSold', 'MiscFeature', 'Fence', 'PoolQC', 'PoolArea', 'PavedDrive', 'GarageCars', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Fireplaces', 'Functional', 'TotRmsAbvGrd', 'KitchenAbvGr', 'BedroomAbvGr', 'HalfBath', 'FullBath', 'BsmtHalfBath', 'BsmtFullBath', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior2nd', 'Exterior1st', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'Heating', 'CentralAir', 'Electrical']
        Xscale = data[z]
        Xscale = pd.DataFrame(transformer.transform(Xscale))
        Xencode = data[q]
        X = Xscale.merge(Xencode.reset_index(), left_index=True, right_index=True)
        return X
    X = scale(X, transformer)

    def encode(data):
        data = pd.get_dummies(data=data)
        return pd.DataFrame(data)
    X = encode(X)

    def re_name(data):
        z = ['MedNhbdBY', 'pca_1', 'pca_2', 'MedNhbdArea', 'BackyardSF', 'PorchSF', '1stFlrSF', '2ndFlrSF', 'AllSF', 'Area', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'TotalBsmtSF']
        column_indices = list(range(0, len(z)))
        new_names = z
        old_names = data.columns[column_indices]
        data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        data.drop(columns='index', inplace=True)
        return data
    X = re_name(X)

    def feat_engin(data):
        data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        data['hasfireplace'] = data['Fireplaces'].apply(
            lambda x: 1 if x > 0 else 0)
        return data
    X = feat_engin(X)

    def reindex_predict(data, cols, model):
        data = data.reindex(columns=cols, fill_value=0)
        data = data[cols]
        data = pd.DataFrame(np.expm1(model.predict(data)))
        return data
    X = reindex_predict(X, cols, model)
    X = "{:,}".format(int(X[0]))
    return render_template('houseprediction.html', prediction=X)

@app.route("/airline")
def airline():
    return render_template('airline.html')

@app.route("/loan")
def loan():
    return render_template('loan.html')

@app.route("/recommend")
def recommend():
    return render_template('recommend.html')

@app.route("/stock")
def stock():
    return render_template('stock.html')
