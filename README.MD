# Laporan Proyek Machine Learning
### Nama : Arul Insan
### Nim : 211351028
### Kelas : IF Pagi B

## Domain Proyek
Proyek ini berkaitan dengan prediksi penyewaan rumah di India. Prediksi ini menjadi penting dalam industri real estate di India yang berkembang pesat. Latar belakang proyek ini adalah untuk membantu pemilik properti dan calon penyewa dalam membuat keputusan yang lebih baik terkait harga penyewaan dan fasilitas rumah.

## Business Understanding 
Dalam konteks ini dapat memberikan wawasan kepada calon penyewa tentang faktor-faktor apa yang mempengaruhi harga sewa rumah di India. Hal ini dapat membantu mereka dalam mengambil keputusan yang lebih terinformasi.

### Problem Statements

Latar belakang masalah:

- Penyewa sulit memprediksi harga sewa rumah yang ada di india dari beberapa faktor yang ada.

- Penyewa tidak dapat mengetahui harga sewa rumah yang ada di beberapa kota India

### Goals

Tujuan dari pernyataan masalah :
- Membuat penelitian memprediksi harga sewa rumah yang ada di beberapa kota india dengan beberapa faktor yang ada

- Penyewa dapat mengetahui harga sewa rumah di india dengan sesuai kebutuhannya


## Data Understanding

Dataset di ambil dari kaggle :

#### [*House Rent Prediction Dataset*](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

Mengimpor library yang di butuhkan :

``` Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
```

Membaca Data CSV
``` Python
df = pd.read_csv('House_Rent_Dataset.csv')
```

Melihat isi data 
``` Python
df.head()
```

Melihat Type yang digunakan pada data pada setiap kolom
```Python
df.info()
```

Melihat apakah data nilainya ada yang Null atau tidak
```Python
sns.heatmap(df.isnull())
```

Mengetahui deskripsi data

``` python

df.describe()
```

Mencari baris data yang duplikat,lalu di hilangkan , dan menunjukan hasil
setelah di hilangkan/di drop

``` python

df.duplicated().sum()

df= df.drop_duplicates()

df.shape
```

### Variabel-variabel pada House Rent Prediction Dataset adalah sebagai berikut:

- Posted On : Merupakan Tanggal di sebarkannya informasi tempat rental (Dtype : Object) \[Contoh : 12-07-2023\]
- BHK : Merupakan singkatan dari Bedroom, Hall, Kitchen yaitu jumlah dari BHK tersebut (Dtype : Int64) \[Contoh : 3 BHK, 2 BHK\]
- Rent : Merupakan harga rental Mata uang Rupee yaitu di daerah india itu sendiri (Dtype : Int64)  \[Contoh : 3000, 4000, 15000 \]
- Size : Size disini yaitu ukuran rumah dari satuan meter persegi atau (sq. ft) (Dtype : Int64)  \[Contoh : 1000 sq.ft, 2000 sq.ft, 3000 sq.ft\]
- floor : Yaitu punya berapa lantai pada rumah tersebut (Dtype : Object) \[Contoh : 1 out of 3, 1 out of 2\]
- Area Type : Yaitu berbagai macam tipe area yang ada di rumah (Dtype : Object)\[Contoh : Super Area, Carpet Area\]
- Area Locality : Yaitu Nama Area Local yang ada di kota itu (Dtype : Object)\[Contoh : Phool Bagan, Kankurgachi, dll\]
- city : Yaitu list kota kota yang ada rental rumah nya (Dtype : Object)\[Contoh : Kolkata, Mumbai\]
- Furnishing Status : Yaitu status perabot yang ada di dalam rumahnya (Dtype : Object)\[Contoh : Berperabot, Semi perabot, dan tanpa perabot\]
- Tenant Preferred : Yaitu List status penyewa yang dibolehkan (Dtype : Object) \[Contoh : lajang, Lajang/keluarga, dan keluarga\]
- Bathroom : Yaitu jumlah kamar mandi yang ada di rumahnya (Dtype : Int64)\[Contoh : 2, 3\]
- Point of contact : Yaitu kontak yang bisa di hubungi (Dtype : Object)\[Contoh : Contact Owner\]


Melihat perbedaan rental di berbagai kota berdasarkan BHK

``` python
plt.figure(figsize = (10, 5))

plt.title("Perbedaan rental di berbagai kota berdasarkan BHK (Bedroom, Hall, Kitchen)")

sns.barplot(x= df["City"], y=df["Rent"], hue= df["BHK"],palette="Set3", ci = None)
``` 
Melihat Perbedaan Rental di berbagai Kota Berdasarkan Tipe Area
``` Python
plt.figure(figsize = (15, 5))

plt.title("Perbedaan Rental di berbagai Kota Berdasarkan Tipe Area")

sns.barplot(x= df["City"], y=df["Rent"], hue= df["Area Type"], palette="Blues",ci = None)
```

Perbedaan Rental Rumah di berbagai kota Berdasarkan Status Perabotnya
``` Python
plt.figure(figsize=(10,5))

plt.title("Perbedaan Rental Rumah di berbagai kota Berdasarkan Status Perabotnya")

sns.barplot(x=df["City"], y=df["Rent"], hue=df["Furnishing Status"], ci=None)
```

Melihat Perbedaan Rental di kota berdasarkan Ukuran rumah
```Python
figure2 = px.bar(df, x=df["City"], y = df["Rent"], color = df["Size"], title="Sewa di Berbagai Kota Berdasarkan Ukuran Perabot Rumah")
figure2.show()
```

```Python
cities = df["City"].value_counts()
labels = cities.index
counts = cities.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=labels, values=counts, hole=0.5)])
fig.update_layout(title_text='Rumah yang masih tersedia untuk di sewa')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=25,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
```

## Data Preperation
Dalam dataset ini untuk menyiapkan data untuk dijadikan model, di perlukan konversi data kategorikal ke numerikal. langkah-langkah yang perlu dilakukan adalah : 
1. Memilih kolom/ Atribut data yang dibutuhkan atau berguna untuk memprediksi harganya
2. Menkonversi isi kolom tipe data kategorikal ke tipe data numerikal

## Mengubah Data Type Object Menjadi Integer

Kolom yang saya pilih dan menurut saya penting yait kolom BHK, Size, Area Type, City, Furnishing Status, Tenant Preferred, dan bathroom

dari kolom tersebut ada kolom yang bertipe object dan harus mengubahnya diantaranya kolom Area Type, city, furnishing status, tenant preferred

```Python
df["Area Type"] = df["Area Type"].map({"Super Area": 1, 
                                           "Carpet Area": 2, 
                                           "Built Area": 3})
df["City"] = df["City"].map({"Mumbai": 4000, "Chennai": 6000, 
                                 "Bangalore": 5600, "Hyderabad": 5000, 
                                 "Delhi": 1100, "Kolkata": 7000})
df["Furnishing Status"] = df["Furnishing Status"].map({"Unfurnished": 0, 
                                                           "Semi-Furnished": 1, 
                                                           "Furnished": 2})
df["Tenant Preferred"] = df["Tenant Preferred"].map({"Bachelors/Family": 2, 
                                                         "Bachelors": 1, 
                                                         "Family": 3})
```

Note : Di dalam kolom kota, di isi dengan pin kode di setiap kota nya
mumbai = 4000, chennai = 6000, dst.

## Modeling
Library untuk modeling sudah di import di atas, yaitu train_test_split dari library sklearn dan Mengimpor decission tree regressor

karena kolom duplicate tidak termasuk kolom yang di perlukan maka kita tidak harus menggunakan kolom tersebut.

Membuat variable untuk menampung beberapa data yang dibutuhkan X sebagai Label dan Y sebagai dependen karena rent dipakai untuk memprediksi, dan lakukan split data untuk data train dan data test (saya masukan data test 10% dan data train 75%)

``` python
x = np.array(df[["BHK", "Size", "Area Type", "City", 
                   "Furnishing Status", "Tenant Preferred", 
                   "Bathroom"]])
y = np.array(df[["Rent"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.10,random_state=42)
```

Menyingkatkan Decision Tree Regressor ke dalam clf dan mebuat variable predikisinya

``` python
clf = DecisionTreeRegressor()
clf.fit(x,y)
pred = clf.predict(xtest)
```
Menampilkan hasil akurasi score menggunakan r2_score

``` python
print('Score Akurasi = ', r2_score(ytest,pred))
```

dari hasil prediksi menghasilkan akurasi 98% yang dimana akurasi tersebut sangat bagus sekali.
atau outputnya :

Score Akurasi =  0.9883176394507799

dibuatkan program contoh visual untuk prediksi harga nya
``` python
features = np.array([[1, 100, 2, 4000, 2, 1, 1]])
print("Prediksi harga rumah dalam dollar = ", clf.predict(features))
``` 

dari hasil di atas menghasilkan nilai 16000, yang dimana harga prediksi di kota tersebut ada sekitar 16000 rupee

## Evaluasi
R-Squared adalah sebuah metrik yang digunakan dalam analisis regresi untuk mengukur sejauh mana model regresi cocok dengan data yang di amati.

Decission Tree Regressor merupakan salah satu jenis algoritma machine learning yang digunakan untuk pemodelan regresi, decision tree regressor digunakan ketika kita ingin memprediksi variabel target yang bersifat kontinu atau numerik

Plotly merupakan library yang dugunakan visualisasi data interaktif dalam bahasa pemrograman python.


``` python
from sklearn.metrics import r2_score
score = r2_score(ytest, pred)  

print(f"Precision = {score}")
```
dari metode yang saya gunakan untuk membuat proyek ini dibutuhkan hal hal di atas ini dan menghasilkan score 98% pada dataset yang saya gunakan

Score Akurasi =  0.9883176394507799

## Deployment

## [*Prediksi Streamlit Link*](https://app-rent-house9arulinsan.streamlit.app/)
