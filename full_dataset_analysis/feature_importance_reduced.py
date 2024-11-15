#!/usr/bin/env python3

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import import_helper as ih
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import export_graphviz
#import graphviz
from sklearn.model_selection import learning_curve
import numpy as np

def load_data(pickle_file):
    """Load data from a pickle file."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    print("Metadata:", data["metadata"])
    print("Number of voxel entries:", len(data["voxel_data"]))
    full_voxel_df = pd.DataFrame(data["voxel_data"])
    return full_voxel_df

def prepare_data(full_voxel_df):
    """Prepare the data by dropping unnecessary columns and expanding lists into separate columns."""
    voxel_df = full_voxel_df.copy()
    voxel_df = voxel_df.drop(columns=["voxel_key", "dataset", "dominant_proportion"])
    plane_coef_df = pd.DataFrame(
        voxel_df["plane_coefficients"].tolist(),
        columns=["plane_coef_1", "plane_coef_2", "plane_coef_3", "plane_coef_4"],
    )
    variance_ratio_df = pd.DataFrame(
        voxel_df["variance_ratios"].tolist(),
        columns=["variance_ratio_1", "variance_ratio_2", "variance_ratio_3"],
    )
    voxel_df = voxel_df.join(plane_coef_df).drop(columns=["plane_coefficients"])
    voxel_df = voxel_df.join(variance_ratio_df).drop(columns=["variance_ratios"])
    return voxel_df

def map_labels_to_categories(label_mapping, label_key):
    """Map label keys to category indices."""
    obstacle_labels = {
        "forest",
        "hedge",
        "building",
        "tree_crown",
        "bush",
        "obstacle",
        "fence",
        "crops",
        "ego_vehicle",
        "wall",
        "debris",
        "bridge",
        "leaves",
        
    }
    if label_key not in label_mapping:
        return 6
    if label_mapping[label_key] in obstacle_labels:
        return 0
    elif label_mapping[label_key] in {"sidewalk", "curb", "cobble", "gravel", "soil", "asphalt"}:
        return 1
    elif label_mapping[label_key] == "low_grass":
        return 2
    elif label_mapping[label_key] == "high_grass":
        return 3
    elif label_mapping[label_key] == "snow":
        return 4
    else:
        #print(f"Unknown label: {label_key}")
        return 5

def map_labels(voxel_df):
    """Map labels in the dataframe to categories."""
    metadata_map = ih.get_metadata_map(ih.load_label_metadata())
    voxel_df["target"] = voxel_df["dominant_label"].apply(
        lambda x: map_labels_to_categories(metadata_map, x)
    )
    return voxel_df

def reduce_features(voxel_df):
    """Reduce the dataset to the specified top features."""
    selected_features = [
        "avg_intensity",
        "elevation_range",
        "plane_coef_3",
        "RMSE",
        "std_intensity",
        "flatness",
        "elongation",
        "target",
        "dominant_label",
    ]
    reduced_voxel_df = voxel_df[selected_features]
    return reduced_voxel_df

def train_and_evaluate(reduced_voxel_df, output_dir):
    """Train the Random Forest model on the reduced dataset, evaluate it, and plot confusion matrix with seaborn heatmap."""
    # Sample down the dataset to 20% of its original size for faster computation
    sampled_df = reduced_voxel_df.sample(frac=1.0, random_state=42)

    # Prepare data
    X = sampled_df.drop(columns=["target", "dominant_label"])
    y = sampled_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    forest_clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    forest_clf.fit(X_train, y_train)

    # Predictions and model evaluation
    y_pred = forest_clf.predict(X_test)

    # Basic accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Detailed classification report
    report = classification_report(y_test, y_pred, zero_division=0)

    # Additional metrics: F1-score, Precision, Recall
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(y_test, y_pred, average="macro")

    # Display results
    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("Macro F1 Score:", f1_macro)
    print("Weighted F1 Score:", f1_weighted)
    print("Macro Precision:", precision_macro)
    print("Macro Recall:", recall_macro)

    # Plot confusion matrix using seaborn heatmap
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    category_labels = {
        0: 'Obstacle',
        1: 'Flat Surface',
        2: 'Low Grass',
        3: 'High Grass',
        4: 'Snow',
        5: 'Other',
        6: 'Not Labelled'
    }

    labels_names = [category_labels[label] for label in labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap='Blues',
        xticklabels=labels_names,
        yticklabels=labels_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Plot feature importances
    feature_importances = pd.Series(
        forest_clf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"))
    plt.close()
    
    f1_per_class = f1_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    
    # Create positions for the bars
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, f1_per_class, width, label='F1 Score', color='skyblue')
    ax.bar(x + width/2, recall_per_class, width, label='Recall', color='lightcoral')
    
    ax.set_ylabel('Score')
    ax.set_title('F1 and Recall Scores by Class')
    ax.set_xticks(x)
    ax.set_xticklabels([category_labels[label] for label in labels], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_recall_scores.png"))
    plt.close()

    # 2. Plot Learning Curve
    train_sizes = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    train_sizes, train_scores, val_scores = learning_curve(
        forest_clf, X, y,
        train_sizes=train_sizes,
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.xlabel('Training Examples')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"))
    plt.close()
    # TODO : Discuss if we want to keep this (need some debugging)
    '''
    # 3. Visualize a single tree from the forest
    # Get a tree from the forest
    estimator = forest_clf.estimators_[0]
    
    # Export the tree to a dot file
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        feature_names=X.columns,
        class_names=[category_labels[i] for i in sorted(category_labels.keys())],
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3  # Limit depth for visibility
    )
    
    # Create and save the tree visualization
    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(output_dir, "decision_tree"), format="png", cleanup=True)
    '''

def main():
    """Main function to execute the data processing and model training."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Output directory relative to the script directory
    output_dir = os.path.join(script_dir, "reduced_feature_model.local")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle_file = os.path.join(
        script_dir, "processed_voxel_data.local", "all_datasets_voxel_data.pkl"
    )

    full_voxel_df = load_data(pickle_file)
    voxel_df = prepare_data(full_voxel_df)
    voxel_df = map_labels(voxel_df)
    reduced_voxel_df = reduce_features(voxel_df)

    train_and_evaluate(reduced_voxel_df, output_dir)

if __name__ == "__main__":
    main()
