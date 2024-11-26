import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Sidebar menu options
menu_option = st.sidebar.selectbox("Choose an Option",
                                   ["Overall Analysis", "Individual Crop Analysis", "Crop Suggestion","Crop Comparison"])

# Overall Analysis
if menu_option == "Overall Analysis":
    st.header("Overall Analysis")

    # Total Crops and List
    st.subheader("Crop Overview")

    # Display the total number of crops
    st.write(f"### Total Number of Crops: {df['label'].nunique()}")

    # Create a DataFrame for the list of crops with indexing starting from 1
    crop_list = pd.DataFrame({'Crop': df['label'].unique()})
    crop_list.index += 1  # Adjust index to start from 1

    # Display the list of crops as a table
    st.write("### List of All Crops")
    st.table(crop_list)

    # Display summary statistics
    st.subheader("Summary Statistics for Each Feature")
    st.write(df.describe().T)

    # Enhanced Box Plot Analysis
    st.subheader("Box Plot Analysis of Each Feature Across Crops")
    features = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']

    # Define colors for each feature plot for distinct visual separation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Set up the plot grid with spacing adjustments for easier viewing
    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for idx, feature in enumerate(features):
        sns.boxplot(data=df, y=feature, x='label', ax=axs[idx // 2, idx % 2], color=colors[idx])
        axs[idx // 2, idx % 2].set_title(f'Distribution of {feature.capitalize()} across Crops', fontsize=12)
        axs[idx // 2, idx % 2].set_xticklabels(axs[idx // 2, idx % 2].get_xticklabels(), rotation=90, fontsize=8)
        axs[idx // 2, idx % 2].set_xlabel("Crop", fontsize=10)
        axs[idx // 2, idx % 2].set_ylabel(feature.capitalize(), fontsize=10)
    axs[3, 1].axis('off')  # Hide the last empty subplot if needed

    st.pyplot(fig)

    # Overall Correlation Matrix
    st.subheader("Correlation Matrix of All Features")
    corr_matrix = df[['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("Animated Visualization: Feature Trends Across Crops")

    # Create a long-format DataFrame for animation with all features
    long_data = df.melt(
        id_vars='label',
        value_vars=['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall'],
        var_name='Feature',
        value_name='Value'
    )

    # Create an animated bar chart
    fig = px.bar(
        long_data,
        x='Value',
        y='label',
        color='Feature',
        animation_frame='Feature',
        orientation='h',
        title="Feature Trends Across Crops (Animated by Feature Type)",
        labels={'label': 'Crop', 'Value': 'Feature Value'},
        template="plotly",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Increase the animation duration (15 seconds for all features)
    frame_duration = 15000 // len(long_data['Feature'].unique())  # Equal distribution of total time
    fig.update_layout(
        height=700,
        width=900,
        xaxis_title="Feature Value",
        yaxis_title="Crops",
        showlegend=True,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": frame_duration}, "transition": {"duration": 800}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0}, "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Add smooth transitions for aesthetic effects
    fig.update_traces(marker_line_width=0.5)
    fig.update_yaxes(categoryorder="total ascending")

    st.plotly_chart(fig)

    # 3D Bar Charts for Average Feature Consumption
    st.subheader("3D Bar Charts for Average Feature Consumption Across Crops")
    features = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']

    for feature in features:
        # Calculate average values per crop
        avg_values = df.groupby('label')[feature].mean().reset_index()
        title = f"Average {feature.capitalize()} required by Crop"

        # Plot 3D bar chart
        fig = px.bar(avg_values, x='label', y=feature, color=feature,
                     title=title, labels={'label': 'Crop', feature: f'Avg {feature.capitalize()}'},
                     height=500, width=800)
        fig.update_layout(
            scene=dict(xaxis=dict(title='Crop'), yaxis=dict(title=f"Avg {feature.capitalize()}")),
            template="plotly"
        )
        st.plotly_chart(fig)


# Individual Crop Analysis
elif menu_option == "Individual Crop Analysis":
    # Sidebar for selecting crop
    crop_list = df['label'].unique()
    selected_crop = st.sidebar.selectbox("Select a Crop", crop_list)

    # Display Crop Name and Overview
    st.title(f"Analysis of {selected_crop}")

    # Filter data for selected crop
    crop_data = df[df['label'] == selected_crop]

    # Recommended Fertilizer Section
    st.header("Recommended Fertilizer")
    st.write("Max, Min, and Avg values for Nitrogen (N), Phosphorus (P), and Potassium (K):")

    # Display max, min, and average values for N, P, and K side by side
    cols = st.columns(3)
    for idx, nutrient in enumerate(['N', 'P', 'K']):
        with cols[idx]:
            st.subheader(f"{nutrient}")
            st.write("Max:", round(crop_data[nutrient].max(), 2))
            st.write("Min:", round(crop_data[nutrient].min(), 2))
            st.write("Avg:", round(crop_data[nutrient].mean()))

    # Optimum Weather Section
    st.header("Optimum Weather Conditions")
    st.write("Max, Min, and Avg values for Temperature, pH, Humidity, and Rainfall:")

    # Display max, min, and average values for temperature, pH, humidity, and rainfall side by side
    cols = st.columns(4)
    for idx, feature in enumerate(['ph', 'humidity','temperature' ,'rainfall']):
        with cols[idx]:
            st.subheader(f"{feature.capitalize()}")
            st.write("Max:", round(crop_data[feature].max()))
            st.write("Min:", round(crop_data[feature].min()))
            st.write("Avg:", round(crop_data[feature].mean()))

    # Statistical Plots Section
    st.header("Statistical Plots")

    # Generate box plots for selected crop's features
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    sns.boxplot(y=crop_data['temperature'], ax=axs[0, 0])
    axs[0, 0].set_title(f'Box Plot of Temperature for {selected_crop}')
    sns.boxplot(y=crop_data['ph'], ax=axs[0, 1])
    axs[0, 1].set_title(f'Box Plot of pH for {selected_crop}')
    sns.boxplot(y=crop_data['humidity'], ax=axs[1, 0])
    axs[1, 0].set_title(f'Box Plot of Humidity for {selected_crop}')
    sns.boxplot(y=crop_data['rainfall'], ax=axs[1, 1])
    axs[1, 1].set_title(f'Box Plot of Rainfall for {selected_crop}')
    st.pyplot(fig)

    # General Description Section
    st.header("General Description for Selected Crop")

    # Correlation Matrix specific to selected crop
    st.subheader("Correlation Matrix")
    corr_matrix = crop_data[['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # Textual Description Section
    st.header("Textual Description for Selected Crop")

    # Analyzing data distribution for specific crop's general description
    description = ""
    for feature, feature_name in zip(['ph', 'rainfall', 'humidity'], ['pH', 'Rainfall', 'Humidity']):
        q1 = crop_data[feature].quantile(0.1)
        q9 = crop_data[feature].quantile(0.9)
        if q9 <= 5.5:
            description += f"- **Low {feature_name} crop**: The majority of data falls under low {feature_name}.\n"
        elif q1 >= 7.5:
            description += f"- **High {feature_name} crop**: The majority of data falls under high {feature_name}.\n"
        else:
            description += f"- **Medium {feature_name} crop**: The majority of data falls under medium {feature_name}.\n"

    st.write(description)

# Crop Suggestion
X = df[['temperature', 'ph', 'humidity', 'rainfall']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit Section
if menu_option == "Crop Suggestion":
    st.header("Crop Suggestion")
    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Input sliders for environmental data
    st.write("### Enter Environmental Conditions:")
    temperature = st.slider("Temperature (°C)", float(X['temperature'].min()), float(X['temperature'].max()), float(X['temperature'].mean()))
    ph = st.slider("pH", float(X['ph'].min()), float(X['ph'].max()), float(X['ph'].mean()))
    humidity = st.slider("Humidity (%)", float(X['humidity'].min()), float(X['humidity'].max()), float(X['humidity'].mean()))
    rainfall = st.slider("Rainfall (mm)", float(X['rainfall'].min()), float(X['rainfall'].max()), float(X['rainfall'].mean()))

    # Predict button
    if st.button("Suggest Suitable Crop"):
        # Predict the most suitable crop
        input_features = np.array([[temperature, ph, humidity, rainfall]])
        predicted_crop = model.predict(input_features)[0]

        # Filter similar crops from the dataset
        similar_crops = df[
            (df['temperature'] >= temperature - 5) & (df['temperature'] <= temperature + 5) &
            (df['ph'] >= ph - 1) & (df['ph'] <= ph + 1) &
            (df['humidity'] >= humidity - 5) & (df['humidity'] <= humidity + 5) &
            (df['rainfall'] >= rainfall - 20) & (df['rainfall'] <= rainfall + 20)
        ]['label'].unique()

        # Display the predicted crop
        st.success(f"The most recommended crop for these conditions is: **{predicted_crop}**")

        # Calculate average N, P, K values for the recommended crop
        crop_data = df[df['label'] == predicted_crop]
        avg_n = round(crop_data['N'].mean())
        avg_p = round(crop_data['P'].mean())
        avg_k = round(crop_data['K'].mean())

        # Display fertilizer recommendation
        st.write("### Fertilizer Recommendation:")
        st.info(f"Recommended fertilizer values for **{predicted_crop}**: \n"
                f"- Nitrogen (N): {avg_n:.2f} \n"
                f"- Phosphorus (P): {avg_p:.2f} \n"
                f"- Potassium (K): {avg_k:.2f}")

        # Display other suitable crops
        if len(similar_crops) > 1:
            st.write("### Other crops that might also be suitable for these conditions:")
            for crop in similar_crops:
                if crop != predicted_crop:
                    st.write(f"- {crop}")
        else:
            st.info("No other crops are suitable for these conditions.")


elif menu_option == "Crop Comparison":
    st.header("Crop Feature Comparison")

    # Sidebar for selecting two crops
    crop_list = df['label'].unique()
    crop_1 = st.selectbox("Select First Crop", crop_list)
    crop_2 = st.selectbox("Select Second Crop", crop_list)

    # Extract data for the selected crops
    numeric_features = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']
    crop_1_data = df[df['label'] == crop_1][numeric_features].mean()
    crop_2_data = df[df['label'] == crop_2][numeric_features].mean()

    # Create radar chart traces for both crops
    fig = go.Figure()

    # Add trace for crop 1 with red color
    fig.add_trace(go.Scatterpolar(
        r=[crop_1_data[feature] for feature in numeric_features],
        theta=numeric_features,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)',  # Red fill with transparency
        line=dict(color='red'),  # Red border line
        name=crop_1
    ))

    # Add trace for crop 2 with black color
    fig.add_trace(go.Scatterpolar(
        r=[crop_2_data[feature] for feature in numeric_features],
        theta=numeric_features,
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.3)',  # Black fill with transparency
        line=dict(color='blue'),  # Black border line
        name=crop_2
    ))

    # Customize layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(df[numeric_features].max())])
        ),
        showlegend=True,
        title="Radar Chart for Crop Feature Comparison"
    )

    # Display the radar chart
    st.plotly_chart(fig)

