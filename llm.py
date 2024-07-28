import pandas as pd
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Load the resume data
resume_df = pd.read_csv('resume.csv')

# Use the 'Resume_str' column for embedding
resume_df['combined_text'] = resume_df['Resume_str']

# Load a pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(resume_df['Category'].unique()))

# Tokenize the text data
inputs = tokenizer(resume_df['combined_text'].tolist(), return_tensors='tf', padding=True, truncation=True, max_length=128)

# Convert categories to numerical labels
label_map = {category: idx for idx, category in enumerate(resume_df['Category'].unique())}
labels = resume_df['Category'].map(label_map).values

# Ensure labels are in the correct format
labels = tf.convert_to_tensor(labels, dtype=tf.int32)

# Prepare dataset
dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), labels))
dataset = dataset.shuffle(100).batch(8).prefetch(tf.data.AUTOTUNE)

# Compile the model with a suitable loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=1)

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_tokenizer')

print("Fine-tuning complete and model saved.")

