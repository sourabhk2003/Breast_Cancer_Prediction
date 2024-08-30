# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 00:24:01 2024

@author: shrut
"""

import numpy as np
import pickle
import streamlit as st
import sklearn 


loaded_model = pickle.load(open('C:/Users/shrut/breast cancer/trained_model.sav', 'rb'))

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

def b_prediction(input_data):
    
    loaded_model = pickle.load(open('C:/Users/shrut/breast cancer/trained_model.sav', 'rb'))
    #input_M  = (19.17,24.8,132.4,1123,0.0974,0.2458,0.2065,0.1118,0.2397,0.078,0.9555,3.568,11.07,116.2,0.003139,0.08297,0.0889,0.0409,0.04484,0.01284,20.96,29.94,151.7,1332,0.1037,0.3903,0.3639,0.1767,0.3176,0.1023)

    
# change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    c = 'The Breast cancer is Malignant'
    d = 'The Breast Cancer is Benign'
    if (prediction[0] == 0):
      return c
    
    else:
      return d
      
    
 
def main():
    st.title("Breast prediction model")
    st.subheader("If result is Benign it means tumor is harmless || if result is Malignant it shows cancer cells are present")
    
    st.write("Enter the dimensions of below features from the imaging report")
        
    mean_radius = st.number_input("enter val  mean_radius")
    mean_texture = st.number_input("enter val mean_texture")
    mean_perimeter = st.number_input("enter val  mean_perimeter")
    mean_area = st.number_input("enter val mean_area")
    mean_smoothness =st.number_input("enter val mean_smoothness")
    mean_compactness = st.number_input("enter val mean_compactness")
    mean_concavity =st.number_input("enter val  mean_concavity")
    mean_concave_points =st.number_input("enter val mean_concave_points")
    mean_symmetry =st.number_input("enter val mean_symmetry")
    mean_fractal_dimension =st.number_input("enter val mean_fractal_dimension")
    radius_error =st.number_input("enter val radius_error")
    texture_error =st.number_input("enter val texture_error")
    perimeter_error =st.number_input("enter val perimeter_error")
    area_error =st.number_input("enter val  area_error")
    smoothness_error =st.number_input("enter val smoothness_error")
    compactness_error =st.number_input("enter val compactness_error")
    concavity_error =st.number_input("enter val concavity_error")
    concave_points_error =st.number_input("enter val concave_points_error")
    symmetry_error =st.number_input("enter val symmetry_error")
    fractal_dimension_error =st.number_input("enter val fractal_dimension_error")
    worst_radius =st.number_input("enter val  worst_radius")
    worst_texture =st.number_input("enter val worst_texture")
    worst_perimeter =st.number_input("enter val  worst_perimeter")
    worst_area = st.number_input("enter val worst_area")
    worst_smoothness = st.number_input("enter val worst_smoothness")
    worst_compactness = st.number_input("enter val worst_compactness")
    worst_concavity = st.number_input("enter val worst_concavity")
    worst_concave_points =st.number_input("enter val  worst_concave_points")
    worst_symmetry = st.number_input("enter val  worst_symmetry")
    worst_fractal_dimension = st.number_input("enter val worst_fractal_dimension")
    
    feature = (mean_radius, mean_texture , mean_perimeter, mean_area ,mean_smoothness, mean_compactness, mean_concavity , mean_concave_points , mean_symmetry, mean_fractal_dimension,radius_error, texture_error, perimeter_error, area_error,smoothness_error, compactness_error, concavity_error,concave_points_error, symmetry_error,fractal_dimension_error, worst_radius, worst_texture,worst_perimeter, worst_area, worst_smoothness , worst_compactness , worst_concavity , worst_concave_points , worst_symmetry , worst_fractal_dimension ) 
     
    
    # creating a button for Prediction
    diagnosis = ""
    if st.button('Test Result'):
         diagnosis = b_prediction(feature)
         
        
    st.info(diagnosis)   
    
        
if __name__ == '__main__':
    main()
    
    
    
    
    
 
