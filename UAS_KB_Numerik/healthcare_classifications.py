import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Membaca dataset
data_path = 'dataset/healthcare_dataset.csv'
data = pd.read_csv(data_path)

print(data.head())

# Label encoding untuk kolom bertipe objek
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

# Memisahkan fitur dan target
X = data.drop('Medical Condition', axis=1)  # Fitur
y = data['Medical Condition']  # Target

# Membagi data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Naive Bayes
clf = GaussianNB()

# Melatih model
clf.fit(X_train, y_train)

# Membuat prediksi pada data pengujian
y_pred = clf.predict(X_test)

# Mengevaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Menyimpan model dan label encoders
# joblib.dump(clf, 'naive_bayes_model.pkl')
# joblib.dump(label_encoders, 'label_encoders.pkl')
