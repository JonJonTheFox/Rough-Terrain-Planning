#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.tree import export_graphviz
import numpy as np
import pickle
import import_helper as ih

global METADATA_MAP
METADATA_MAP = ih.get_metadata_map(ih.load_label_metadata())

def load_data(pickle_file):
    """Load data from a pickle file."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    full_voxel_df = pd.DataFrame(data["voxel_data"])
    return full_voxel_df

def prepare_data(voxel_df):
    """Prepare the data by keeping only the necessary columns."""
    required_columns = [
        "dataset", "voxel_key", "center_x", "center_y", "num_points",
        "dominant_label", "dominant_proportion", "avg_intensity",
        "elevation_range", "plane_coef_3", "RMSE", "flatness", "elongation"
    ]
    voxel_df = voxel_df[required_columns]
    return voxel_df

def map_labels(voxel_df):
    """Map labels to categories."""
    obstacle_labels = {
        "forest", "hedge", "building", "tree_crown", "bush", "obstacle",
        "fence", "crops", "ego_vehicle", "wall", "debris", "bridge", "leaves"
    }
    surface_labels = {"sidewalk", "curb", "cobble", "gravel", "soil", "asphalt"}
    
    def label_mapper(int_label):
        try:
            label = METADATA_MAP[int_label]
        except KeyError:
            return 5
        if label in obstacle_labels:
            return 0
        elif label in surface_labels:
            return 1
        elif label == "low_grass":
            return 2
        elif label == "high_grass":
            return 3
        elif label == "snow":
            return 4
        else:
            return 5
    
    voxel_df["target"] = voxel_df["dominant_label"].apply(label_mapper)
    return voxel_df

def train_and_evaluate(voxel_df, output_dir):
    """Train and evaluate a Random Forest model."""
    X = voxel_df.drop(columns=["target", "dominant_label", "voxel_key", "dataset", "center_x", "center_y", "dominant_proportion"])
    y = voxel_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    forest_clf.fit(X_train, y_train)
    y_pred = forest_clf.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    label_names = ['Obstacle', 'Surface', 'Low Grass', 'High Grass', 'Snow', 'Unknown']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


    # Feature importances
    feature_importances = pd.Series(forest_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    feature_importances.plot(kind="bar", figsize=(10, 6))
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"))
    plt.close()

    # F1 and Recall scores
    f1_per_class = f1_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    labels = sorted(y_test.unique())

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, f1_per_class, width, label='F1 Score', color='skyblue')
    plt.bar(x + width/2, recall_per_class, width, label='Recall', color='lightcoral')
    plt.xticks(x, label_names, rotation=45)
    plt.xlabel("Class Labels")
    plt.ylabel("Scores")
    plt.title("F1 and Recall Scores by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_recall_scores.png"))
    plt.close()


    # Learning curve
    train_sizes = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    train_sizes, train_scores, val_scores = learning_curve(
        forest_clf, X, y, train_sizes=train_sizes, cv=5, scoring="f1_weighted", n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    plt.plot(train_sizes, val_mean, label="Cross-validation score", color="red")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="red")
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"))
    plt.close()


def main():
    """Main function to load data, preprocess, and train the model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "reduced_feature_model.local")
    os.makedirs(output_dir, exist_ok=True)

    pickle_file = os.path.join(script_dir, "processed_voxel_data_reduced.local/all_datasets_voxel_data.pkl")
    full_voxel_df = load_data(pickle_file)
    voxel_df = prepare_data(full_voxel_df)
    voxel_df = map_labels(voxel_df)
    train_and_evaluate(voxel_df, output_dir)

if __name__ == "__main__":
    main()
