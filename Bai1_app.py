# File: bai1_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ── Tiêu đề ứng dụng ──
st.title('Dự đoán giá xe ô tô cũ')
st.markdown('Nhập thông tin xe để nhận dự đoán giá từ mô hình RandomForest.')


# ── Thanh bên: nhập dữ liệu ──
st.sidebar.header('Thông tin xe cần dự đoán')
engine = st.sidebar.slider('Dung tích động cơ (L)', 0.5, 8.0, 2.0, step=0.1)
mileage = st.sidebar.number_input('Số km đã đi', min_value=0, max_value=500000, value=50000)
age     = st.sidebar.slider('Tuổi xe (năm)', 0, 20, 3)


# ── Huấn luyện mô hình (dùng dữ liệu cố định) ──
@st.cache_resource   # cache để không train lại mỗi lần
def train_model():
    data = {'Engine_size':[2.1,6.0,3.0,2.0,2.0,4.0,1.0,3.0,4.0,0.0,2.0,2.0],
            'Mileage':[77656,23094,51652,132611,106544,375611,813687,
                       655332,300225,102850,450000,250000],
            'Age':[3,0,1,1,1,3,6,3,0,0,1,0],
            'Price':[11818,60235,16521,26134,21106,43756,11813,
                     67655,22300,22500,45000,10285]}
    df = pd.DataFrame(data)
    X  = df[['Engine_size','Mileage','Age']]
    y  = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()


# ── Dự đoán và hiển thị ──
X_input  = pd.DataFrame([[engine, mileage, age]],
                         columns=['Engine_size','Mileage','Age'])
y_pred   = model.predict(X_input)[0]


st.metric(label='Giá dự đoán', value=f'${y_pred:,.0f}')


# ── Biểu đồ Feature Importance ──
st.subheader('Mức độ ảnh hưởng của từng đặc trưng')
fig, ax = plt.subplots()
ax.barh(['Engine_size','Mileage','Age'], model.feature_importances_, color='steelblue')
ax.set_xlabel('Importance')
st.pyplot(fig)
