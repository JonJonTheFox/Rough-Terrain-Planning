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
        "animal",
        "barrel",
        "barrier_tape",
        "bicycle",
        "boom_barrier",
        "bridge",
        "building",
        "bus",
        "car",
        "caravan",
        "container",
        "debris",
        "fence",
        "guard_rail",
        "heavy_machinery",
        "hedge",
        "kick_scooter",
        "misc_sign",
        "motorcycle",
        "obstacle",
        "person",
        "pole",
        "rail_track",
        "rider",
        "road_block",
        "rock",
        "scenery_vegetation",
        "street_light",
        "traffic_cone",
        "traffic_light",
        "traffic_sign",
        "trailer",
        "tree_crown",
        "tree_root",
        "tree_trunk",
        "truck",
        "tunnel",
        "wall",
        "wire",
    }
    passable_labels = {"asphalt", "cobble", "gravel", "sidewalk", "soil", "low_grass"}
    if label_key not in label_mapping:
        return 8
    if label_mapping[label_key] in obstacle_labels:
        return 0
    elif label_mapping[label_key] == "cobble":
        return 1
    elif label_mapping[label_key] == "gravel":
        return 2
    elif label_mapping[label_key] == "sidewalk":
        return 3
    elif label_mapping[label_key] == "soil":
        return 4
    elif label_mapping[label_key] == "high_grass":
        return 5
    elif label_mapping[label_key] == "low_grass":
        return 6
    else:
        return 7

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
    sampled_df = reduced_voxel_df.sample(frac=0.20, random_state=42)

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
        1: 'Cobble',
        2: 'Gravel',
        3: 'Sidewalk',
        4: 'Soil',
        5: 'High Grass',
        6: 'Low Grass',
        7: 'Other',
        8: 'Unknown',
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
