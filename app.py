import gradio as gr
import joblib
import pandas as pd
import numpy as np
import spaces
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. DEKLARASI CUSTOM TRANSFORMER CLASS (WAJIB SAMA PERSIS DENGAN DI NOTEBOOK)
# ==============================================================================
class IQROutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_cap):
        self.columns_to_cap = columns_to_cap
        self.limits_ = {}

    def fit(self, X, y=None):
        for col in self.columns_to_cap:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.limits_[col] = {
                    'lower': q1 - 1.5 * iqr,
                    'upper': q3 + 1.5 * iqr
                }
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in self.columns_to_cap:
            if col in X_transformed.columns and col in self.limits_:
                upper_limit = self.limits_[col]['upper']
                lower_limit = self.limits_[col]['lower']
                X_transformed[col] = np.where(X_transformed[col] > upper_limit, upper_limit, X_transformed[col])
                X_transformed[col] = np.where(X_transformed[col] < lower_limit, lower_limit, X_transformed[col])
        return X_transformed

# ==============================================================================
# 2. LOAD PIPELINE INFERENSI
# ==============================================================================
try:
    model_pipeline = joblib.load('logreg_pipeline_olist.pkl')
except Exception as e:
    model_pipeline = None
    print(f"Gagal memuat model. Pastikan 'logreg_pipeline_olist.pkl' berada di folder yang sama. Error: {e}")

# ==============================================================================
# 3. FUNGSI PREDIKSI
# ==============================================================================
@spaces.GPU
def predict_churn(frequency, monetary, payment_installments, freight_ratio, avg_items_per_order,
                 payment_type, customer_region, purchased_approved, purchased_delivered,
                 delivered_estimated, has_canceled_order, review_score, product_weight_g, product_volume_cm3):
    
    if model_pipeline is None:
        return "Error", "Model tidak ditemukan atau gagal dimuat."
        
    has_canceled_order_val = 1 if has_canceled_order == "Ya" else 0
    
    # Koordinat Geolocation default Brazil
    geolocation_lat = -23.55
    geolocation_lng = -46.63
    
    # Inisialisasi dictionary dengan semua kolom yang diharapkan oleh model (order harus tepat)
    input_data = {
        'frequency': frequency,
        'purchased_approved': purchased_approved,
        'delivered_estimated': delivered_estimated,
        'purchased_delivered': purchased_delivered,
        'product_weight_g': product_weight_g,
        'product_volume_cm3': product_volume_cm3,
        'geolocation_lat': geolocation_lat,
        'geolocation_lng': geolocation_lng,
        'payment_installments': payment_installments,
        'monetary': monetary,
        'review_score': review_score,
        'has_canceled_order': has_canceled_order_val,
        'avg_items_per_order': avg_items_per_order,
        'freight_ratio': freight_ratio,
        
        # Default categorical dummies ke 0 (karena memakai drop_first=True)
        'customer_state_northeastern': 0,
        'customer_state_northern': 0,
        'customer_state_southeastern': 0,
        'customer_state_southern': 0,
        'payment_type_credit_card': 0,
        'payment_type_debit_card': 0,
        'payment_type_voucher': 0
    }
    
    # Atur nilai *flag* dummy berdasarkan pilihan *dropdown* user
    if customer_region == "Northeastern": input_data['customer_state_northeastern'] = 1
    elif customer_region == "Northern": input_data['customer_state_northern'] = 1
    elif customer_region == "Southern": input_data['customer_state_southern'] = 1
    elif customer_region == "Southeastern": input_data['customer_state_southeastern'] = 1
    
    if payment_type == "Credit Card": input_data['payment_type_credit_card'] = 1
    elif payment_type == "Debit Card": input_data['payment_type_debit_card'] = 1
    elif payment_type == "Voucher": input_data['payment_type_voucher'] = 1

    # Konversi ke DataFrame dengan susunan kolom persis seperti matriks X di training
    X_inference = pd.DataFrame([input_data])
    
    # Prediksi menggunakan pipeline utuh
    proba = model_pipeline.predict_proba(X_inference)[0][1]
    prediction = model_pipeline.predict(X_inference)[0]
    
    # Format Status
    if prediction == 1:
        status = "⚠️ RISIKO TINGGI (Pelanggan diprediksi akan Churn)"
    else:
        status = "✅ AMAN (Pelanggan diprediksi akan tetap aktif)"
        
    return f"{proba * 100:.2f}%", status

# ==============================================================================
# 4. INTERMUKA PENGGUNA (GRADIO UI/UX)
# ==============================================================================
with gr.Blocks(title="Olist Churn Predictor") as demo:
    gr.Markdown("# 📊 Aplikasi Prediksi Churn Pelanggan - Olist E-Commerce")
    gr.Markdown("Masukkan data metrik pelanggan di bawah ini untuk memprediksi probabilitas *churn*.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🛒 Transaksi & Metrik Keuangan")
            frequency = gr.Number(label="Frequency (Jumlah Order Unik)", value=1, minimum=1)
            monetary = gr.Number(label="Monetary (Total Pengeluaran dalam R$)", value=150.0, minimum=0.0)
            payment_installments = gr.Number(label="Rata-rata Cicilan Pembayaran", value=1, minimum=1)
            freight_ratio = gr.Slider(label="Freight Ratio (Rasio Ongkir)", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
            avg_items_per_order = gr.Number(label="Rata-rata Jumlah Item per Order", value=1.0, minimum=1.0)
            
            gr.Markdown("### 💳 Metode Pembayaran & Lokasi")
            payment_type = gr.Dropdown(label="Metode Pembayaran Utama", choices=["Boleto", "Credit Card", "Debit Card", "Voucher"], value="Credit Card")
            customer_region = gr.Dropdown(label="Wilayah Asal Pelanggan (Region)", choices=["Southeastern", "Southern", "Centralwestern", "Northeastern", "Northern"], value="Southeastern")
            
        with gr.Column():
            gr.Markdown("### 📦 Pengalaman Pengiriman (Logistik)")
            purchased_approved = gr.Number(label="Waktu Pembelian hingga Disetujui (Jam)", value=2.0, minimum=0.0)
            purchased_delivered = gr.Number(label="Waktu Pembelian hingga Paket Tiba (Hari)", value=12, minimum=0)
            delivered_estimated = gr.Number(label="Selisih Hari Tiba vs Estimasi (Positif=Cepat, Negatif=Lambat)", value=5)
            has_canceled_order = gr.Dropdown(label="Pernah Memiliki Riwayat Pesanan Dibatalkan?", choices=["Tidak", "Ya"], value="Tidak")
            
            gr.Markdown("### ⭐ Karakteristik Produk & Kepuasan")
            review_score = gr.Slider(label="Rata-rata Skor Ulasan (Review Score)", minimum=1.0, maximum=5.0, value=4.5, step=0.1)
            product_weight_g = gr.Number(label="Rata-rata Berat Produk (Gram)", value=1500.0, minimum=0.0)
            product_volume_cm3 = gr.Number(label="Rata-rata Volume Produk (cm³)", value=5000.0, minimum=0.0)

    btn = gr.Button("🔮 Hitung Probabilitas Churn", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("### Hasil Analisis Risiko:")
    with gr.Row():
        out_proba = gr.Textbox(label="Probabilitas Churn", scale=1)
        out_status = gr.Textbox(label="Status", scale=2)
        
    btn.click(fn=predict_churn, 
              inputs=[frequency, monetary, payment_installments, freight_ratio, avg_items_per_order,
                      payment_type, customer_region, purchased_approved, purchased_delivered,
                      delivered_estimated, has_canceled_order, review_score, product_weight_g, product_volume_cm3],
              outputs=[out_proba, out_status])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())