import pickle
import streamlit as st

model = pickle.load(open('estimasi_harga_rental.sav','rb'))

st.title('Prediksi Rental Rumah di berbagai kota India')
bhk = st.number_input('Jumlah BHK (Bedroom, Hall, Kitchen)', min_value= 1)
size = st.slider('Ukuran Rumah (Sq. ft) ',min_value=1000,max_value=8000)
select_area_type = st.selectbox('Pilih Jenis Area',['Super Area','Carpet Area','Built Area'])

if select_area_type == 'Super Area':
    area_type = 1
elif select_area_type == 'Carpet Area':
    area_type = 2
else:
    area_type = 3  
    
select_kota = st.selectbox('Pilih Kota',['Mumbai','Chennai','Bangalore','Hyderabad','Dehli','Kolkata'])

if select_kota == 'Mumbai':
    kota = 4000
elif select_kota == 'Chennai':
    kota = 6000
elif select_kota=='Bangalore':
    kota = 5600
elif select_kota =='Hyderabad':
    kota = 5000
elif select_kota == 'Dehli':
    kota = 1100
else:
    kota = 7000
    
select_furnishing = st.selectbox('Pilih Status Perabotan',['Tanpa Perabot','Semi-Perabot','Berperabot'])

if select_furnishing == 'Tanpa Perabot':
    furnishing = 0
elif select_furnishing == 'Semi-Perabot' :
    furnishing = 1
else :
    furnishing = 2
select_tenant = st.selectbox('Status Penghuni',['Lajang','Lajang/Keluarga','Keluarga'])

if select_tenant =='Lajang':
    tenant = 1
elif select_tenant == 'Keluarga':
    tenant = 3
else:
    tenant = 2
bathroom = st.number_input('jumlah kamar mandi',min_value=1)

if st.button('Prediksi'):
    predict = model.predict(
        [[bhk,size,area_type,kota,furnishing,tenant,bathroom]]
    )
    st.success(f'Prediksi Harga Rental (Rupee): {predict}')