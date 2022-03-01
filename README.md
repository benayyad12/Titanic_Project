# Titanic_Project
Let's discover if you will survive or not on Titanic ! using ML


  ___For Dataset__ :

[https://github.com/datasciencedojo/datasets/blob/f0ccab6a7ceafdff780052166fb6fab3311398eb/titanic.csv]

## Handling Mising values : 
 - replacing null values with : mean or mode it depends on the situation 
```python
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
```

```python
df['Age']=df['Age'].fillna(df['Age'].mean())
```

## Encode cathegorical variables :

```python
from sklearn.preprocessing import LabelEncoder
```
```python
Label_=LabelEncoder()
```

```python
df['Sex']=Label_.fit_transform(df['Sex'])

```

## Features extraction : 
  - Correlation matrix 
```python
correlation_matrix = df.corr()
```

```python
figure = plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix,annot=True)
plt.show()
```

## help Function to calculate accuracy on train/test : 

```python
def model_(name_model):    
    model = name_model()
    model.fit(x_train_vec,y_train)
    prediction_train = model.predict(x_train_vec)
    accuracy_train = accuracy_score(y_train,prediction_train)
    print('model : ',name_model, 'Accuracy on train : ',accuracy_train)
    
    
    prediction_test = model.predict(x_test_vec)
    accuracy_test = accuracy_score(y_test,prediction_test)
    print('model : ',name_model, 'Accuracy on test : ',accuracy_test)
```

    
