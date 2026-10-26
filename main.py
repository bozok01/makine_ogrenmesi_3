# 1. KÜTÜPHANELER
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 2. AYLIK İŞKUR EXCEL DOSYALARINI OKUMA
# Her dosya bir ay sonundaki engelli istihdam kontenjanı durumunu gösterir.
dosyalar = glob.glob("veriler/*.xlsx")
# 14 sayı sütunu: 7 ölçümün kamu ve özel sektör değerleri.
sutunlar = [
    "Isyeri_Kamu", "Isyeri_Ozel",
    "Zorunlu_Sayi_Kamu", "Zorunlu_Sayi_Ozel",
    "50ustu_Calisan_Kamu", "50ustu_Calisan_Ozel",
    "Acik_Kontenjan_Kamu", "Acik_Kontenjan_Ozel",
    "50alti_Calisan_Kamu", "50alti_Calisan_Ozel",
    "Fazla_Kontenjan_Kamu", "Fazla_Kontenjan_Ozel",
    "Toplam_Calisan_Kamu", "Toplam_Calisan_Ozel"
]

veri_listesi = []
for dosya in dosyalar:
    eslesme = re.match(r"(\d{1,2})-.*-(\d{4})\.xlsx$", os.path.basename(dosya), re.IGNORECASE)
    if eslesme is None:
        continue
    ay = int(eslesme.group(1))
    yil = int(eslesme.group(2))
    excel = pd.read_excel(dosya, sheet_name=0, header=None, usecols="A:O", nrows=85)

    # İlk üç satır başlık; son il satırından sonra genel toplam ve açıklama var.
    il_verisi = excel.iloc[3:84, :].copy()
    il_verisi.columns = ["Il"] + sutunlar
    il_verisi = il_verisi.dropna(subset=["Il"])
    il_verisi["Il"] = il_verisi["Il"].astype(str).str.strip()
    il_verisi = il_verisi[~il_verisi["Il"].str.contains("GENEL TOPLAM", case=False)]
    for sutun in sutunlar:
        il_verisi[sutun] = pd.to_numeric(il_verisi[sutun], errors="coerce")
    il_verisi = il_verisi.dropna(subset=sutunlar)
    il_verisi["Yil"] = yil
    il_verisi["Ay"] = ay
    il_verisi["Tarih"] = pd.Timestamp(yil, ay, 1)
    il_verisi["Acik_Kontenjan"] = il_verisi["Acik_Kontenjan_Kamu"] + il_verisi["Acik_Kontenjan_Ozel"]
    veri_listesi.append(il_verisi)

if not veri_listesi:
    raise FileNotFoundError("Ay-yıl bilgili İŞKUR Excel dosyaları bulunamadı.")

veriler = pd.concat(veri_listesi, ignore_index=True)
veriler = veriler.sort_values(["Il", "Tarih"]).reset_index(drop=True)
print("Dosya sayısı:", len(veri_listesi))
print("Aylık il kayıtları:", len(veriler))
print(veriler[["Il", "Yil", "Ay", "Acik_Kontenjan"]].head())

os.makedirs("sonuclar/aylik_iskur", exist_ok=True)
veriler.to_csv("sonuclar/aylik_iskur/aylik_veri.csv", index=False, encoding="utf-8-sig")

# 3. AYLIK AÇIK KONTENJAN ÖZETİ
aylik_ozet = veriler.groupby("Tarih")["Acik_Kontenjan"].sum().reset_index()
aylik_ozet.to_csv("sonuclar/aylik_iskur/aylik_toplam.csv", index=False, encoding="utf-8-sig")

son_tarih = veriler["Tarih"].max()
son_ay_il = veriler[veriler["Tarih"] == son_tarih].sort_values("Acik_Kontenjan", ascending=False)
son_ay_il[["Il", "Acik_Kontenjan", "Acik_Kontenjan_Kamu", "Acik_Kontenjan_Ozel"]].to_csv(
    "sonuclar/aylik_iskur/son_ay_il_ozeti.csv", index=False, encoding="utf-8-sig"
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(aylik_ozet["Tarih"], aylik_ozet["Acik_Kontenjan"], marker="o", color="teal")
ax.set_title("İŞKUR Engelli Açık Kontenjanlarının Aylara Göre Değişimi")
ax.set_xlabel("Ay")
ax.set_ylabel("Açık kontenjan sayısı (kamu + özel)")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("sonuclar/aylik_iskur/aylik_kontenjan_grafigi.png", dpi=150)
plt.close(fig)

# 4. SONRAKİ AYDAKİ ARTIŞI SINIFLANDIRMA
# Hedef, her ilde açık kontenjanın sonraki ay artıp artmadığıdır.
veriler["Sonraki_Ay"] = veriler.groupby("Il")["Tarih"].shift(-1)
veriler["Sonraki_Acik_Kontenjan"] = veriler.groupby("Il")["Acik_Kontenjan"].shift(-1)
beklenen_ay = veriler["Tarih"] + pd.offsets.MonthBegin(1)
# Eksik bir ay varsa onu atlayıp iki ay sonrasını yanlışlıkla "sonraki ay" sayma.
eksik_ay = veriler["Sonraki_Ay"] != beklenen_ay
veriler.loc[eksik_ay, ["Sonraki_Ay", "Sonraki_Acik_Kontenjan"]] = np.nan
model_verisi = veriler.dropna(subset=["Sonraki_Ay"]).copy()
model_verisi["Yil"] = model_verisi["Tarih"].dt.year
model_verisi["Ay"] = model_verisi["Tarih"].dt.month

sayisal_sutunlar = sutunlar + ["Acik_Kontenjan", "Yil", "Ay"]
x = model_verisi[["Il"] + sayisal_sutunlar]

# Zaman sıralı %70 eğitim / %30 test: sonraki aylar testte kalır.
test_aylari = sorted(model_verisi["Sonraki_Ay"].unique())
test_baslangici = test_aylari[int(len(test_aylari) * 0.70)]
egitim = model_verisi[model_verisi["Sonraki_Ay"] < test_baslangici]
test = model_verisi[model_verisi["Sonraki_Ay"] >= test_baslangici]
x_egitim = x.loc[egitim.index]
x_test = x.loc[test.index]
# Eğitim kümesindeki medyanı eşik alıp hedefi düşük/yüksek olarak sınıflandır.
medyan_kontenjan = egitim["Sonraki_Acik_Kontenjan"].median()
y_egitim = np.where(egitim["Sonraki_Acik_Kontenjan"] > medyan_kontenjan, "Yüksek", "Düşük")
y_test = np.where(test["Sonraki_Acik_Kontenjan"] > medyan_kontenjan, "Yüksek", "Düşük")
coğunluk_baslangic = pd.Series(y_test).value_counts(normalize=True).max()

def yeni_on_isleme():
    return ColumnTransformer([
        ("il", OneHotEncoder(handle_unknown="ignore"), ["Il"]),
        ("sayisal", StandardScaler(), sayisal_sutunlar)
    ])

modeller = [
    make_pipeline(yeni_on_isleme(), LogisticRegression(max_iter=5000, class_weight="balanced")),
    make_pipeline(yeni_on_isleme(), DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)),
    make_pipeline(yeni_on_isleme(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
    make_pipeline(yeni_on_isleme(), SVC(kernel="rbf", class_weight="balanced", C=2))
]
model_isimleri = ["Lojistik Regresyon", "Karar Ağacı", "KNN", "SVM"]
siniflar = ["Düşük", "Yüksek"]

# 5. HER MODELİ AYRI AYRI DEĞERLENDİRME
rapor_yolu = "sonuclar/aylik_iskur/model_raporlari.txt"
model_dosya_adlari = ["lojistik_regresyon", "karar_agaci", "knn", "svm"]
dogruluklar = []
degerlendirmeler = []

with open(rapor_yolu, "w", encoding="utf-8") as rapor:
    rapor.write(f"Eğitim dönemi: {egitim['Tarih'].min():%Y-%m} - {egitim['Tarih'].max():%Y-%m}\\n")
    rapor.write(f"Test hedef ayları: {test_baslangici:%Y-%m} - {test['Sonraki_Ay'].max():%Y-%m}\\n")
    rapor.write(f"Eğitim kayıt sayısı: {len(egitim)} | Test kayıt sayısı: {len(test)}\\n")
    rapor.write("Hedef: sonraki ay açık kontenjan sınıfı (Düşük / Yüksek).\\n")
    rapor.write(f"Test çoğunluk sınıfı başlangıç doğruluğu: {coğunluk_baslangic:.3f}\\n\\n")

    for isim, dosya_adi, model in zip(model_isimleri, model_dosya_adlari, modeller):
        model.fit(x_egitim, y_egitim)
        tahmin = model.predict(x_test)
        dogruluk = accuracy_score(y_test, tahmin)
        matris = confusion_matrix(y_test, tahmin, labels=siniflar)
        rapor_metin = classification_report(
            y_test, tahmin, labels=siniflar, target_names=siniflar, zero_division=0
        )
        rapor_veri = classification_report(
            y_test, tahmin, labels=siniflar, target_names=siniflar,
            zero_division=0, output_dict=True
        )
        dogruluklar.append(dogruluk)
        degerlendirmeler.append({
            "isim": isim, "dosya_adi": dosya_adi, "dogruluk": dogruluk,
            "tahmin": tahmin, "matris": matris, "rapor_metin": rapor_metin,
            "rapor_veri": rapor_veri
        })

        rapor.write(f"{isim} - doğruluk: {dogruluk:.3f}\\n")
        rapor.write(f"Karmaşıklık matrisi (satırlar gerçek, sütunlar tahmin):\\n{matris}\\n")
        rapor.write(rapor_metin + "\\n\\n")
        with open(f"sonuclar/aylik_iskur/{dosya_adi}_siniflandirma_raporu.txt", "w", encoding="utf-8") as tek_rapor:
            tek_rapor.write(f"{isim}\\nDoğruluk (accuracy): {dogruluk:.3f}\\n\\n")
            tek_rapor.write("Karmaşıklık matrisi (satırlar gerçek, sütunlar tahmin):\\n")
            tek_rapor.write(str(matris) + "\\n\\n")
            tek_rapor.write(rapor_metin)
        print(isim, "doğruluk:", round(dogruluk, 3))

sonuclar = pd.DataFrame({"Model": model_isimleri, "Dogruluk": dogruluklar})
sonuclar.to_csv("sonuclar/aylik_iskur/model_sonuclari.csv", index=False, encoding="utf-8-sig")

# Dört modelin karmaşıklık matrislerini tek görselde karşılaştır.
fig, eksenler = plt.subplots(2, 2, figsize=(10, 8))
for ax, sonuc in zip(eksenler.flat, degerlendirmeler):
    gorunum = ConfusionMatrixDisplay(sonuc["matris"], display_labels=siniflar)
    gorunum.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(sonuc["isim"])
    ax.set_xlabel("Tahmin edilen sınıf")
    ax.set_ylabel("Gerçek sınıf")
fig.tight_layout()
fig.savefig("sonuclar/aylik_iskur/karmasiklik_matrisi_karsilastirmasi.png", dpi=160)
plt.close(fig)

# Her model için accuracy, karmaşıklık matrisi ve sınıflandırma raporu görseli.
for sonuc in degerlendirmeler:
    rapor_veri = sonuc["rapor_veri"]
    satir_adlari = siniflar + ["Macro avg", "Weighted avg"]
    tablo = []
    for ad in satir_adlari:
        anahtar = ad if ad in rapor_veri else ad.lower()
        deger = rapor_veri[anahtar]
        tablo.append([
            ad, f"{deger['precision']:.2f}", f"{deger['recall']:.2f}",
            f"{deger['f1-score']:.2f}", f"{int(deger['support'])}"
        ])

    fig = plt.figure(figsize=(12, 5.8))
    fig.suptitle(f"{sonuc['isim']} — Doğruluk (Accuracy): {sonuc['dogruluk']:.3f}", fontsize=16, weight="bold")
    duzen = fig.add_gridspec(1, 2, width_ratios=[1, 1.55])
    ax_m = fig.add_subplot(duzen[0, 0])
    gorunum = ConfusionMatrixDisplay(sonuc["matris"], display_labels=siniflar)
    gorunum.plot(ax=ax_m, cmap="Blues", colorbar=False, values_format="d")
    ax_m.set_title("Karmaşıklık Matrisi")
    ax_m.set_xlabel("Tahmin edilen sınıf")
    ax_m.set_ylabel("Gerçek sınıf")

    ax_t = fig.add_subplot(duzen[0, 1])
    ax_t.axis("off")
    ax_t.set_title("Sınıflandırma Raporu", pad=14)
    tablo_gorsel = ax_t.table(
        cellText=tablo,
        colLabels=["Sınıf", "Precision", "Recall", "F1-score", "Support"],
        cellLoc="center", loc="center"
    )
    tablo_gorsel.auto_set_font_size(False)
    tablo_gorsel.set_fontsize(10)
    tablo_gorsel.scale(1, 1.8)
    fig.text(0.5, 0.04, f"Test dönemi: {test_baslangici:%Y-%m} – {test['Sonraki_Ay'].max():%Y-%m} | Test kayıt sayısı: {len(test)}", ha="center")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(f"sonuclar/aylik_iskur/{sonuc['dosya_adi']}_degerlendirme.png", dpi=160)
    plt.close(fig)

# Doğruluk tablosunu hem CSV hem de görsel olarak kaydet.
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.axis("off")
ax.set_title("Modellerin Doğruluk (Accuracy) Tablosu", pad=18, fontsize=14)
tablo_gorsel = ax.table(
    cellText=[[isim, f"{deger:.3f}", f"%{deger * 100:.1f}"] for isim, deger in zip(model_isimleri, dogruluklar)],
    colLabels=["Model", "Accuracy", "Doğruluk (%)"],
    cellLoc="center", loc="center"
)
tablo_gorsel.auto_set_font_size(False)
tablo_gorsel.set_fontsize(11)
tablo_gorsel.scale(1, 1.7)
fig.tight_layout()
fig.savefig("sonuclar/aylik_iskur/dogruluk_tablosu.png", dpi=160)
plt.close(fig)

# Modellerin doğruluk skorlarını karşılaştıran grafik.
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(model_isimleri, dogruluklar, color=["#2a9d8f", "#457b9d", "#e9c46a", "#6a4c93"])
ax.bar_label(bars, labels=[f"{deger:.3f}" for deger in dogruluklar], padding=4)
ax.set_ylim(0, 1)
ax.set_ylabel("Doğruluk (Accuracy)")
ax.set_title("Modellerin Doğruluk Skorlarının Karşılaştırılması")
ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("sonuclar/aylik_iskur/model_dogruluklari.png", dpi=160)
plt.close(fig)

# 5. KISA SONUÇ RAPORU
with open("sonuclar/aylik_iskur/analiz_ozeti.txt", "w", encoding="utf-8") as dosya:
    dosya.write("İŞKUR AYLIK ENGELLİ KONTENJAN ANALİZİ\n")
    dosya.write(f"Aylık dosya sayısı: {len(veri_listesi)}\n")
    dosya.write(f"Dönem: {veriler['Tarih'].min():%Y-%m} - {veriler['Tarih'].max():%Y-%m}\n")
    dosya.write(f"İl-ay kayıt sayısı: {len(veriler)}\n")
    dosya.write("2025 Mayıs dosyasında bir ilin bazı değerleri eksik olduğu için ilgili geçiş model eğitiminden çıkarıldı.\n")
    dosya.write(f"Test çoğunluk sınıfı başlangıç doğruluğu: {coğunluk_baslangic:.3f}\n")
    dosya.write(f"Tahmin hedefi: sonraki ay açık kontenjanın düşük/yüksek olması (eğitim medyanı: {medyan_kontenjan:.0f}).\n")
    dosya.write("Modeller kesin kontenjan sayısını değil, il için düşük/yüksek sınıfını tahmin eder.\n\n")
    dosya.write("Aylara göre Türkiye toplam açık kontenjanı:\n")
    dosya.write(aylik_ozet.to_string(index=False) + "\n\n")
    dosya.write("Model doğrulukları:\n" + sonuclar.to_string(index=False) + "\n")

print("Analiz dosyaları sonuclar/aylik_iskur klasörüne kaydedildi.")
