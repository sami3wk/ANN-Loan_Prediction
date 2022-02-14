def replacer(df):
    import pandas as pd
    Q = pd.DataFrame(df.isna().sum())
    Q.columns=["CT"]
    w = list(Q[Q.CT>0].index)
    cat = []
    con = []
    for i in w:
        if(df[i].dtypes=="object"):
            cat.append(i)
        else:
            con.append(i)
    for i in con:
        replacer = df[i].mean()
        df[i] = df[i].fillna(replacer)
    for i in cat:
        replacer = pd.DataFrame(df[i].value_counts()).index[0]
        df[i] = df[i].fillna(replacer)
    return df
       

    
def preprocessing(df,removalcolumns,ycolumn):
    import pandas as pd
    df = df.drop(labels=removalcolumns,axis=1)
    X = df.drop(labels=ycolumn,axis=1)
    Y = df[ycolumn]

    cat = []
    con = []
    for i in X.columns:
        if(X[i].dtypes=="object"):
            cat.append(i)
        else:
            con.append(i)

    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    X1 = pd.DataFrame(mm.fit_transform(X[con]),columns=con)
    X2 = pd.get_dummies(X[cat])
    X = X1.join(X2)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y = pd.DataFrame(le.fit_transform(Y),columns=Y.columns)

    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=30)
    return xtrain,xtest,ytrain,ytest

