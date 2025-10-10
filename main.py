import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# ----------------------
# 1. Settings
# ----------------------
csv_path = "data/esoc_name_file.csv"        # your metadata file
img_dir = "data/insects"         # folder with species subfolders (ScientificName style)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TARGET_LEVEL = "order"           # <-- choose: "Order", "Family", "Genus", "Species"

# ----------------------
# 2. Load taxonomy CSV
# ----------------------
df = pd.read_csv(csv_path, encoding="latin1")

# Use "ScientificName" column to match folder names
df["species_folder"] = df["scientific_name"].str.strip()

# Build mappings
species_to_order  = dict(zip(df["species_folder"], df["order"]))
species_to_family = dict(zip(df["species_folder"], df["family"]))
species_to_genus  = dict(zip(df["species_folder"], df["genus"]))
species_to_species = dict(zip(df["species_folder"], df["species_folder"]))  # identity

mappings = {
    "order": species_to_order,
    "family": species_to_family,
    "genus": species_to_genus,
    "species": species_to_species
}

#
# remove bad data
#
import shutil

# List all valid species folders from your CSV
valid_species = set(df["species_folder"].str.strip())

# List all folders in your image directory
all_folders = os.listdir(img_dir)

# Remove any folders not in CSV
for folder in all_folders:
    if folder not in valid_species:
        full_path = os.path.join(img_dir, folder)
        if os.path.isdir(full_path):
            print(f"Removing folder not in CSV: {folder}")
            # Optionally delete or move the folder
            # shutil.rmtree(full_path)  # delete
            # OR move to a backup location
            shutil.move(full_path, "backup/data")

# ----------------------
# 3. Load dataset (folders = ScientificName)
# ----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    img_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"   # force 3 channels
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    img_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"   # force 3 channels
)

train_ds = train_ds.ignore_errors()
val_ds = val_ds.ignore_errors()

species_classes = train_ds.class_names
print("Species classes (folders):", len(species_classes))

def ensure_rgb(x, y):
    if x.shape[-1] == 1:  # grayscale
        x = tf.image.grayscale_to_rgb(x)
    return x, y

# ----------------------
# 4. Remap labels to chosen level
# ----------------------
# def remap_labels(dataset, class_names, mapping):
#     class_to_idx = {name: idx for idx, name in enumerate(class_names)}
#     idx_to_higher = {class_to_idx[s]: mapping[s] for s in class_names if s in mapping}

#     unique_labels = sorted(set(idx_to_higher.values()))
#     label_to_new_idx = {label: i for i, label in enumerate(unique_labels)}

#     def convert(x, y):
#         higher_label = idx_to_higher[int(y.numpy())]
#         return x, label_to_new_idx[higher_label]

#     def tf_convert(x, y):
#         x, y = tf.py_function(convert, [x, y], [tf.float32, tf.int64])
#         x.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
#         y.set_shape(())
#         return x, y

#     return dataset.map(tf_convert), unique_labels

# train_ds, class_names = remap_labels(train_ds, species_classes, mappings[TARGET_LEVEL])
# val_ds, _ = remap_labels(val_ds, species_classes, mappings[TARGET_LEVEL])

# num_classes = len(class_names)
# print(f"Training at level: {TARGET_LEVEL}, Classes: {num_classes}")

# ----------------------
# 4. Remap labels to chosen level (TF-friendly)
# ----------------------
def remap_labels_tf(dataset, class_names, mapping):
    """
    Remap the species-level labels to a higher taxonomy level (order/family/genus)
    without losing the batch dimension.
    """
    # Map species folder -> higher level label
    class_to_higher = {cls: mapping[cls] for cls in class_names if cls in mapping}
    
    # Get all unique higher-level labels
    unique_labels = sorted(set(class_to_higher.values()))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    # Create a TF lookup table for species -> higher-level index
    keys = tf.constant(list(class_to_higher.keys()))
    values = tf.constant([label_to_idx[class_to_higher[k]] for k in class_to_higher])
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1
    )

    # Function to map species index -> higher-level index
    def map_labels(x, y):
        # y is species index; convert to string class_name first
        y_str = tf.gather(class_names, y)
        y_new = table.lookup(y_str)
        return x, y_new

    # Apply mapping
    dataset = dataset.map(map_labels, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset, unique_labels

# Apply to train and validation datasets
train_ds, class_names_mapped = remap_labels_tf(train_ds, species_classes, mappings[TARGET_LEVEL])
val_ds, _ = remap_labels_tf(val_ds, species_classes, mappings[TARGET_LEVEL])

num_classes = len(class_names_mapped)
print(f"Training at level: {TARGET_LEVEL}, Classes: {num_classes}")


# ----------------------
# 5. Prefetch & Normalize
# ----------------------
AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

train_ds = train_ds.map(ensure_rgb)
val_ds = val_ds.map(ensure_rgb)

# ----------------------
# 6. Build Model
# ----------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
              
# ----------------------
# 7. Train
# ----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ----------------------
# 8. Save model + classes
# ----------------------
model_name = f"insect_{TARGET_LEVEL.lower()}_classifier.h5"
model.save(model_name)

with open(f"{TARGET_LEVEL.lower()}_classes.txt", "w") as f:
    f.write("\n".join(class_names))

print(f"✅ Model saved as {model_name}")
