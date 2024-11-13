#!/usr/bin/env python3

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import import_helper as ih
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance


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


def plot_correlation_matrix(voxel_df, output_dir):
    """Plot and save the feature correlation matrix."""
    correlation_matrix = voxel_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation_matrix.png"))
    plt.close()


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


def plot_category_distribution(voxel_df, output_dir):
    """Plot and save the distribution of categories."""
    category_distribution = voxel_df["target"].value_counts().sort_index()
    print(category_distribution)
    plt.figure(figsize=(8, 6))
    plt.bar(category_distribution.index, category_distribution.values)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Distribution of Categories in Voxel Data")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_distribution.png"))
    plt.close()


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    forest_clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    forest_clf.fit(X_train, y_train)
    return forest_clf


def evaluate_model(model, X_test, y_test, output_dir, feature_names, prefix=""):
    """Evaluate the model and save performance metrics and plots."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(y_test, y_pred, average="macro")

    print(f"{prefix}Model Accuracy:", accuracy)
    print(f"\n{prefix}Classification Report:\n", report)
    print(f"{prefix}Macro F1 Score:", f1_macro)
    print(f"{prefix}Weighted F1 Score:", f1_weighted)
    print(f"{prefix}Macro Precision:", precision_macro)
    print(f"{prefix}Macro Recall:", recall_macro)

    # Classification report dataframe
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    category_metrics = report_df.iloc[:-3][["f1-score", "recall"]]

    # Plot F1 scores and recall
    fig, ax = plt.subplots(figsize=(10, 6))
    category_metrics.plot(kind="bar", ax=ax, color=["skyblue", "orange"])
    ax.set_title(f"{prefix}F1 Scores and Recall by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Score")
    ax.legend(["F1 Score", "Recall"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f"{prefix.lower()}f1_scores_and_recall_by_category.png"
        )
    )
    plt.close()

    # Feature importance
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    feature_importances.nlargest(100).plot(
        kind="barh", title=f"{prefix}Top Feature Importances", figsize=(10, 8)
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix.lower()}top_feature_importances.png")
    )
    plt.close()

    # Permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": perm_importance.importances_mean,
        }
    ).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10, 8))
    perm_importance_df.head(20).plot(
        kind="barh", x="Feature", y="Importance", legend=False
    )
    plt.title(f"{prefix}Top 20 Feature Importances (Permutation Importance)")
    plt.xlabel("Mean Decrease in Accuracy")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix.lower()}permutation_importance.png")
    )
    plt.close()


def main():
    """Main function to execute the data processing and model training pipeline."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the output directory relative to the script directory
    output_dir = os.path.join(script_dir, "feature_reduction.local")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle_file = os.path.join(
        script_dir, "processed_voxel_data.local", "all_datasets_voxel_data.pkl"
    )

    # Load and prepare data
    full_voxel_df = load_data(pickle_file)
    voxel_df = prepare_data(full_voxel_df)

    # Plot correlation matrix
    plot_correlation_matrix(voxel_df, output_dir)

    # Map labels to categories
    voxel_df = map_labels(voxel_df)

    # Plot category distribution
    plot_category_distribution(voxel_df, output_dir)

    # Sample down the dataset to 10% of its original size
    sampled_df = voxel_df.sample(frac=0.1, random_state=42)

    # Prepare data for full feature set
    X = sampled_df.drop(columns=["target", "dominant_label"])
    y = sampled_df["target"]
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate model with full features
    model = train_random_forest(X_train, y_train)
    evaluate_model(
        model, X_test, y_test, output_dir, feature_names, prefix="Full Features "
    )

    # Reduce the dataset to the selected features
    selected_features = [
        "avg_intensity",
        "elevation_range",
        "plane_coef_3",
        "RMSE",
        "flatness",
        "elongation",
    ]

    # Prepare data for reduced feature set
    voxel_df_reduced = voxel_df[selected_features + ["target", "dominant_label"]]
    sampled_df_reduced = voxel_df_reduced.sample(frac=0.1, random_state=42)
    X_reduced = sampled_df_reduced.drop(columns=["target", "dominant_label"])
    y_reduced = sampled_df_reduced["target"]
    feature_names_reduced = X_reduced.columns
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
        X_reduced, y_reduced, test_size=0.2, random_state=42
    )

    # Train and evaluate model with reduced features
    model_reduced = train_random_forest(X_train_reduced, y_train_reduced)
    evaluate_model(
        model_reduced,
        X_test_reduced,
        y_test_reduced,
        output_dir,
        feature_names_reduced,
        prefix="Reduced Features ",
    )


if __name__ == "__main__":
    main()
