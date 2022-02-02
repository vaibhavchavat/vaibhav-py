# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
def main():
    activities=['Data Analysis','Plotting','Model Building','About']
    choice=st.sidebar.selectbox("select Menu",activities)
    if choice =='Data Analysis':
        st.subheader('exploratory data analysis')
        datasets=st.file_uploader("Upload a dataset",type=['csv','txt','xlsx'])
        if datasets is not None:
            df=pd.read_csv(datasets)
            st.dataframe(df.head())
            
            if st.checkbox("Show Details no. pf records"):
                st.write(df.shape)
            if st.checkbox('Show Columns'):
                all_columns =df.columns.to_list()
                st.write(all_columns)
            if st.checkbox("Show details"):
                st.write(df.describe())
            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
            if st.checkbox('Co-relational(Matplotlib)'):
                plt.matshow(df.corr())
                st.pyplot( )
            if st.checkbox('Co-relation Plot(Seaborn)'):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
            if st.checkbox('Pie plot'):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox('Select 1 Column', all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
                
    elif choice == 'Plotting':
        st.subheader("plotting for dataset")
        data=st.file_uploader("Upload a dataset",type=['csv','txt','xlsx'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
            if st.checkbox("show column count"):
                st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
                st.pyplot()  
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names )
            if st.button("Generate plot"):
                st.success("Generating customerising plot of{} for{}".format(all_columns_names,type_of_plot))
				
                if type_of_plot=='area':
                    cust_data=df[selected_columns_names ]
                    st.area_chart(cust_data)
                    
                elif type_of_plot=='bar':
                    cust_data=df[selected_columns_names]
                    st.bar_chart(cust_data)
                    
                elif type_of_plot=='line':
                    cust_data=df[selected_columns_names]
                    st.line_chart(cust_data)
                    
                elif type_of_plot:
                    cust_data=df[selected_columns_names].plot(kind=type_of_plot)
                    st.pyplot()
                    
    elif choice == 'Model Building':
        st.subheader("model building for dataset")
        data=st.file_uploader("Upload a dataset",type=['csv','txt','xlsx'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
           
            # model building
            x=df.iloc[:,1:-1]
            y=df.iloc[:,-1]
            
            #model list
            models=[]
            models.append(('LR',LogisticRegression()))
            models.append(('KNN',KNeighborsClassifier()))
            models.append(('Decision Tree',DecisionTreeClassifier()))
            models.append(('NB',GaussianNB()))
            models.append(('SVM',SVC()))
            
            models_name=[]
            all_models=[]
            models_mean=[]
            models_std=[]
            scoring='accuracy'
            for name,model in models:
                kfold=model_selection.KFold(n_splits=10,random_state=None)
                cv_results=model_selection.cross_val_score(model,x,y,cv=kfold,scoring=scoring)
                models_name.append(name)
                models_mean.append(cv_results.mean())
                models_std.append(cv_results.std())
                accuracy_results={"model name":models_name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
                all_models.append(accuracy_results)
            if st.checkbox("Metrics As Table"):
                st.dataframe(pd.DataFrame(zip(models_name,models_mean,models_std),\
                                          columns=["Algo","Mean of Accuracy","Std"]))



    elif choice == 'About':
        st.subheader("all about for dataset")
    
    
    
if __name__ =='__main__':
 main()