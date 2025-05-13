import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Đọc dữ liệu
du_lieu = pd.read_csv(r"C:\Users\Admin\Documents\train.csv")

# ======== PHÂN TÍCH DỮ LIỆU (EDA) ========

print("\n=== Thông tin dữ liệu gốc ===")
print(du_lieu.info())

print("\n=== Mô tả thống kê ===")
print(du_lieu.describe())

# Biểu đồ phân phối giá nhà
plt.figure(figsize=(8, 4))
sns.histplot(du_lieu['SalePrice'], kde=True)
plt.title('Phân phối giá SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.show()

# Heatmap các thuộc tính tương quan mạnh nhất với SalePrice
ma_tran_tuong_quan = du_lieu.corr(numeric_only=True)
top_tuong_quan = ma_tran_tuong_quan['SalePrice'].abs().sort_values(ascending=False)[1:11]
plt.figure(figsize=(10, 6))
sns.heatmap(du_lieu[top_tuong_quan.index.tolist() + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap tương quan với SalePrice (Top 10)")
plt.tight_layout()
plt.show()

# Biểu đồ Boxplot với OverallQual
plt.figure(figsize=(10, 5))
sns.boxplot(x='OverallQual', y='SalePrice', data=du_lieu)
plt.title('SalePrice theo OverallQual')
plt.tight_layout()
plt.show()

# Scatterplot diện tích sống với giá nhà
sns.lmplot(x='GrLivArea', y='SalePrice', data=du_lieu, height=5, aspect=2, line_kws={"color": "red"})
plt.title('GrLivArea vs SalePrice', fontsize=14)
plt.tight_layout()
plt.show()

# ======== TIỀN XỬ LÝ DỮ LIỆU ========

# Chuyển biến phân loại thành biến giả (one-hot encoding)
du_lieu = pd.get_dummies(du_lieu)

# Điền giá trị thiếu
du_lieu = du_lieu.fillna(du_lieu.mean(numeric_only=True))
du_lieu = du_lieu.fillna(0)

# In dữ liệu sau xử lý
print("Dữ liệu sau xử lý:")
print(du_lieu.head())

# ======== TÁCH TẬP DỮ LIỆU ========

# Biến đầu vào và mục tiêu
X = du_lieu.drop(columns=['Id', 'SalePrice'])  # Đặc trưng
y = du_lieu['SalePrice']                       # Biến mục tiêu

# Chia tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======== MÔ HÌNH TUYẾN TÍNH ========
hoi_quy_tuyen_tinh = LinearRegression()
hoi_quy_tuyen_tinh.fit(X_train, y_train)
du_doan_lr = hoi_quy_tuyen_tinh.predict(X_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, du_doan_lr, color='blue', alpha=0.5, label='Dự đoán')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Đường chuẩn')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('Dự đoán giá nhà với hồi quy tuyến tính')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Đánh giá mô hình tuyến tính
print("\n Đánh giá mô hình Hồi quy tuyến tính:")
print("MAE:", mean_absolute_error(y_test, du_doan_lr))
print("MSE:", mean_squared_error(y_test, du_doan_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, du_doan_lr)))
print("R2:", r2_score(y_test, du_doan_lr))

# ======== MÔ HÌNH RANDOM FOREST ========
thong_so = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Khởi tạo mô hình Random Forest
mo_hinh_rf = RandomForestRegressor(random_state=42)

# Khởi tạo GridSearchCV
grid_search = GridSearchCV(estimator=mo_hinh_rf,
                           param_grid=thong_so,
                           cv=3,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1,
                           verbose=1)

# Huấn luyện mô hình với GridSearch
print("Đang tìm tham số tối ưu cho Random Forest...")
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
mo_hinh_tot_nhat = grid_search.best_estimator_
print("Tham số tốt nhất tìm được:", grid_search.best_params_)

# Dự đoán với mô hình tốt nhất
du_doan_toi_uu = mo_hinh_tot_nhat.predict(X_test)

# Đánh giá mô hình sau tối ưu
print("\n Đánh giá mô hình Random Forest tối ưu:")
print("MAE:", mean_absolute_error(y_test, du_doan_toi_uu))
print("MSE:", mean_squared_error(y_test, du_doan_toi_uu))
print("RMSE:", np.sqrt(mean_squared_error(y_test, du_doan_toi_uu)))
print("R2:", r2_score(y_test, du_doan_toi_uu))

# Lưu kết quả ra file CSV
ids = du_lieu.loc[X_test.index, 'Id']
submission = pd.DataFrame({'Id': ids, 'SalePrice': du_doan_toi_uu})
submission.to_csv('submission_toi_uu.csv', index=False)
print(" File dự đoán đã lưu: submission_toi_uu.csv")

# ======== XUẤT FILE SUBMISSION ========
id_du_doan = du_lieu.loc[X_test.index, 'Id']
submission = pd.DataFrame({'Id': id_du_doan, 'SalePrice': du_doan_toi_uu})
submission.to_csv('submission.csv', index=False)
print("\n Kết quả dự đoán đã được lưu vào submission.csv")
print("Hoàn thành!")